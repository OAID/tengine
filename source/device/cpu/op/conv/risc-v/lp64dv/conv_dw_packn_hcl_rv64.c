#include "convolution_param.h"
#include "api/c_api.h"

#include "graph/graph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_module.h"
#include <stdio.h>

extern int conv_dw_packn_kernel_run(const ir_node_t* ir_node, const ir_tensor_t* input_tensor, const ir_tensor_t* filter_tensor, const ir_tensor_t* bias_tensor, ir_tensor_t* output_tensor, const struct conv_priv_info* priv_info, const struct conv_param* params, const int num_thread, const int cpu_affinity);
extern int conv_dw_packn_kernel_prerun(const ir_node_t* ir_node, const ir_tensor_t* input_tensor, const ir_tensor_t* filter_tensor, struct conv_priv_info* info, struct conv_param* params);
extern int conv_dw_packn_kernel_postrun(const ir_node_t* ir_node, struct conv_priv_info* info);

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    const ir_node_t* ir_node = exec_node->ir_node;
    ir_graph_t* ir_graph = ir_node->graph;
    const ir_tensor_t* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    const ir_tensor_t* filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    const ir_tensor_t* bias_tensor = NULL;
    ir_tensor_t* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    const int num_thread = exec_graph->num_thread;
    const int cpu_affinity = exec_graph->cpu_affinity;

    if (ir_node->input_num > 2)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    }

    const struct conv_param* params = (const struct conv_param*)ir_node->op.param_mem;
    const struct conv_priv_info* info = (const struct conv_priv_info*)exec_node->ops_priv;

    if (exec_graph->mode != TENGINE_MODE_FP32)
    {
        return -1;
    }

    return conv_dw_packn_kernel_run(ir_node, input_tensor, filter_tensor, bias_tensor, output_tensor, info, params, num_thread, cpu_affinity);
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* info = sys_malloc(sizeof(struct conv_priv_info));
    if (!info)
    {
        return -1;
    }

    memset(info, 0, sizeof(*info));
    exec_node->ops_priv = info;

    return 0;
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
    struct conv_param* param = (struct conv_param*)ir_node->op.param_mem;
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor;
    struct tensor* output_tensor;

    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;
    int pad_h1 = param->pad_h1;
    int pad_w1 = param->pad_w1;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int in_c = input_tensor->dims[1] / group;
    int out_c = output_tensor->dims[1] / group;
    int outh = output_tensor->dims[2];
    int outw = output_tensor->dims[3];

    if (!(input_tensor->data_type == TENGINE_DT_FP32))
        return 0;

    if (kernel_h != kernel_w || input_tensor->dims[0] > 1)
        return 0;

    if (param->group > 1
        && in_c == 1 && out_c == 1 && pad_h0 == pad_h1 && pad_w0 == pad_w1
        && dilation_h == 1 && dilation_w == 1 && kernel_h == 3 && kernel_w == 3
        && ((stride_h == 1 && stride_w == 1) || (stride_h == 2 && stride_w == 2)))
        return OPS_SCORE_BEST;
    else
        return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    const ir_node_t* ir_node = exec_node->ir_node;
    ir_graph_t* ir_graph = ir_node->graph;
    const ir_tensor_t* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    const ir_tensor_t* filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct conv_priv_info* info = (struct conv_priv_info*)exec_node->ops_priv;

    struct conv_param* params = (struct conv_param*)ir_node->op.param_mem;
    return conv_dw_packn_kernel_prerun(ir_node, input_tensor, filter_tensor, info, params);
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    const ir_node_t* ir_node = exec_node->ir_node;
    struct conv_priv_info* info = (struct conv_priv_info*)exec_node->ops_priv;
    return conv_dw_packn_kernel_postrun(ir_node, info);
}

static struct node_ops hcl_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = NULL,
    .postrun = postrun,
    .init_node = init_node,
    .release_node = release_node,
    .score = score};

int register_conv_dw_packn_hcl_rv64_op()
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

int unregister_conv_dw_packn_hcl_rv64_op()
{
    return unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
}
