#include "api/c_api.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "op/conv/risc-v/lp64dv/vsetvl_rvv.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "operator/op.h"
#include <math.h>
#include "device/cpu/cpu_module.h"

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    const float* input_data = input_tensor->data;
    float* output_data = output_tensor->data;

    const int batch = input_tensor->dims[0];
    const int channel = input_tensor->dims[1];
    const int img_size = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];

    vsetvl_e32_m2();

    for (int b = 0; b < batch; ++b)
    {
        int i = 0;
        for (; i < (img_size & -8); i += 8)
        {
            asm("vle32.v    v0, (%0);\n"
                "vfabs.v    v2, v0;\n"
                "vse32.v    v2, (%1);\n"
                :
                : "r"(input_data), "r"(output_data)
                : "memory");
            input_data += 8;
            output_data += 8;
        }

        for (; i < img_size; ++i)
        {
            *output_data = fabsf(*input_data);
            output_data++;
            input_data++;
        }
    }

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* ir_node)
{
    struct graph* graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(graph, ir_node->input_tensors[0]);
    if (input_tensor->data_type != TENGINE_MODE_FP32 || input_tensor->layout != TENGINE_LAYOUT_NCHW)
    {
        return 0;
    }

    return OPS_SCORE_PREFER;
}

static struct node_ops hcl_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score};

int register_absval_hcl_rv64_op()
{
    return register_builtin_node_ops(OP_ABSVAL, &hcl_node_ops);
}

int unregister_absval_hcl_rv64_op()
{
    return unregister_builtin_node_ops(OP_ABSVAL, &hcl_node_ops);
}
