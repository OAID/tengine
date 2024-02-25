#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "op/conv/risc-v/lp64dv/vsetvl_rvv.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>

struct add_n_op_param
{
    int in_num;
    void** input_data;
};

static int ref_add_n_fp32(const float** input, float* output, int size, const struct add_n_op_param* param)
{
    int in_num = param->in_num;
    vsetvl_e32_m2();

    float* output_data = output;
    int i = 0;
    for (; i < (size & -8); i += 8)
    {
        asm("vmv.v.x  v0, x0;\n");
        int n = 0;
        for (; n < (in_num & -8); n += 8)
        {
            const float** inputs = input + n;
            const float* in0 = inputs[0] + i;
            const float* in1 = inputs[1] + i;
            const float* in2 = inputs[2] + i;
            const float* in3 = inputs[3] + i;
            const float* in4 = inputs[4] + i;
            const float* in5 = inputs[5] + i;
            const float* in6 = inputs[6] + i;
            const float* in7 = inputs[7] + i;

            asm("vle32.v    v2,  (%0);\n"
                "vle32.v    v4,  (%1);\n"
                "vle32.v    v6,  (%2);\n"
                "vle32.v    v8,  (%3);\n"
                "vle32.v    v10, (%4);\n"
                "vle32.v    v12, (%5);\n"
                "vle32.v    v14, (%6);\n"
                "vle32.v    v16, (%7);\n"
                "vfadd.vv   v0, v0, v2;\n"
                "vfadd.vv   v0, v0, v4;\n"
                "vfadd.vv   v0, v0, v6;\n"
                "vfadd.vv   v0, v0, v8;\n"
                "vfadd.vv   v0, v0, v10;\n"
                "vfadd.vv   v0, v0, v12;\n"
                "vfadd.vv   v0, v0, v14;\n"
                "vfadd.vv   v0, v0, v16;\n"
                :
                : "r"(in0), "r"(in1), "r"(in2), "r"(in3), "r"(in4), "r"(in5), "r"(in6), "r"(in7));
        }

        for (; n < in_num; n += 1)
        {
            const float* in0 = input[n] + i;
            asm("vle32.v    v2, (%0);\n"
                "vfadd.vv   v0, v0, v2;\n"
                :
                : "r"(in0));
        }

        asm("vse32.v    v0, (%0);\n"
            :
            : "r"(output_data)
            : "memory");
        output_data += 8;
    }

    for (; i < size; i += 1)
    {
        output[i] = input[0][i];
        for (int n = 1; n < in_num; n++)
        {
            output[i] += input[n][i];
        }
    }

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct add_n_op_param* add_n_op_param = (struct add_n_op_param*)sys_malloc(sizeof(struct add_n_op_param));
    exec_node->ops_priv = add_n_op_param;
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct add_n_op_param* add_n_op_param = (struct add_n_op_param*)exec_node->ops_priv;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    int in_num = ir_node->input_num;
    add_n_op_param->in_num = in_num;
    add_n_op_param->input_data = (void**)sys_malloc(sizeof(void*) * in_num);

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor_a = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    uint32_t elem_num = input_tensor_a->elem_num;
    struct add_n_op_param* add_n_op_param = (struct add_n_op_param*)exec_node->ops_priv;
    for (int i = 0; i < add_n_op_param->in_num; i++)
    {
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        void* data = input_tensor->data;
        add_n_op_param->input_data[i] = data;
    }
    const void** input = (const void**)add_n_op_param->input_data;

    float* output = (float*)output_tensor->data;
    for (uint32_t i = 0; i < elem_num; i++)
    {
        output[i] = 0;
    }
    ref_add_n_fp32((const float**)input, output, elem_num, add_n_op_param);
    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct add_n_op_param* add_n_op_param = (struct add_n_op_param*)exec_node->ops_priv;
    sys_free(add_n_op_param->input_data);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    if (input_tensor->data_type != TENGINE_DT_FP32 || input_tensor->layout != TENGINE_LAYOUT_NCHW)
        return 0;

    return OPS_SCORE_PREFER;
}

static struct node_ops add_n_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = NULL,
    .postrun = postrun,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_add_n_hcl_rv64_op()
{
    return register_builtin_node_ops(OP_ADD_N, &add_n_node_ops);
}

int unregister_add_n_hcl_rv64_op()
{
    return unregister_builtin_node_ops(OP_ADD_N, &add_n_node_ops);
}
