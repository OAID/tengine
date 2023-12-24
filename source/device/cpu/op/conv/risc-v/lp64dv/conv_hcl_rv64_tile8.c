#include "convolution_param.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "operator/op.h"
#include "api/c_api.h"
#include "utility/log.h"
#include "utility/sys_port.h"
#include "device/cpu/cpu_module.h"
#include <string.h>
#include <stdio.h>

extern int conv_hcl_prerun_tile8(struct node* ir_node, struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* output_tensor, struct conv_priv_info* info, struct conv_param* param);
extern int conv_hcl_run_tile8(struct node* ir_node, struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor, struct tensor* output_tensor, struct conv_priv_info* info, struct conv_param* param, int num_thread, int cpu_affinity);
extern int conv_hcl_get_shared_mem_size_rv64_tile8(struct tensor* input_tensor, struct tensor* output_tensor, struct conv_param* param);
extern int conv_hcl_postrun_tile8(struct node* ir_node, struct conv_priv_info* info);

static int init_node(struct node_ops* ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* kernel_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct conv_param* params = ir_node->op.param_mem;
    struct conv_priv_info* info = sys_malloc(sizeof(struct conv_priv_info));
    if (!info)
    {
        return -1;
    }

    memset(info, 0, sizeof(*info));
    exec_node->ops_priv = info;

    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        exec_node->shared_mem_size = conv_hcl_get_shared_mem_size_rv64_tile8(input_tensor, output_tensor, params);
        exec_node->shared_pack4_mem_size = 0;
    }
    else
    {
        TLOG_ERR("Tengine work node %s not support %d\n", ir_node->name, exec_graph->mode);
        return -1;
    }

    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* param = ir_node->op.param_mem;
    struct conv_priv_info* info = exec_node->ops_priv;

    info->cpu_type = exec_graph->cpu_affinity;

    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        if (exec_node->shared_mem_size < exec_graph->shared_mem_size)
        {
            info->external_im2col_mem = 1;
            info->im2col_buffer = exec_graph->shared_mem;
            info->im2col_buffer_size = exec_graph->shared_mem_size;
        }

        if (exec_node->shared_pack4_mem_size < exec_graph->shared_pack4_mem_size)
        {
            info->external_im2col_pack4_mem = 0;
            info->im2col_buffer_pack4 = NULL;
            info->im2col_buffer_pack4_size = 0;
        }

        if (param->group > 1 && param->kernel_h == 7 && param->kernel_w == 7)
        {
            info->external_interleave_pack4_mem = 0;
        }
        else
        {
            info->external_interleave_pack4_mem = 1;
        }

        if (conv_hcl_prerun_tile8(ir_node, input_tensor, filter_tensor, output_tensor, info, param) < 0)
        {
            TLOG_ERR("hcl conv tile8 prerun failed.\n");
            return -1;
        }
    }
    else
    {
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct tensor* bias_tensor = NULL;
    if (ir_node->input_num > 2)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    }

    struct conv_param* params = ir_node->op.param_mem;
    struct conv_priv_info* info = exec_node->ops_priv;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    if (exec_graph->mode == TENGINE_DT_FP32)
    {
        int ret = conv_hcl_run_tile8(ir_node, input_tensor, filter_tensor, bias_tensor, output_tensor, info, params, num_thread, cpu_affinity);
        if (ret < 0)
        {
            TLOG_ERR("conv_hcl_run_tile8 %s run failed: %d\n", ir_node->name, ret);
            return ret;
        }
    }
    else
    {
        TLOG_ERR("Tengine work node %s not support %d mode\n", ir_node->name, exec_graph->mode);
        return -1;
    }

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        return conv_hcl_postrun_tile8(exec_node->ir_node, exec_node->ops_priv);
    }
    else
    {
        TLOG_ERR("Tengine work node %s not support %d mode\n", exec_node->ir_node->name, exec_graph->mode);
        return -1;
    }
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* info = exec_node->ops_priv;
    sys_free(info);
    exec_node->ops_priv = NULL;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* ir_node)
{
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* kernel_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct conv_param* param = ir_node->op.param_mem;

    if (input_tensor->data_type != TENGINE_DT_FP32)
    {
        return 0;
    }

    if (param->group != 1)
    {
        return 0;
    }

    return OPS_SCORE_PREFER;
}
#if 1
static struct node_ops hcl_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = reshape,
    .postrun = postrun,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_conv_hcl_rv64_tile8_op()
{
    TLOG_INFO("register conv_hcl_tile8 op");
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

int unregister_conv_hcl_rv64_tile8_op()
{
    unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
    return 0;
}
#endif
