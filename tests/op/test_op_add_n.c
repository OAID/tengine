#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"

int create_test_add_n_node(graph_t graph, const char* input_node_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    node_t test_node = create_graph_node(graph, node_name, OP_ADD_N_NAME);
    if (NULL == test_node)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    node_t input_node = get_graph_node(graph, input_node_name);
    for (int i = 0; i < get_node_output_number(input_node); ++i)
    {
        tensor_t input_tensor = get_node_output_tensor(input_node, i);
        set_node_input_tensor(test_node, i, input_tensor);
    }

    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    if (!output_tensor)
    {
        fprintf(stderr, "create graph output tensor failed.\n");
        return -1;
    }

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);
    return 0;
}

#define define_test_case(__func, __layout, ...)                                                                           \
    static int __func()                                                                                                   \
    {                                                                                                                     \
        const char* test_node_name = "absval";                                                                            \
        int data_type = TENGINE_DT_FP32;                                                                                  \
        int layout = __layout;                                                                                            \
        int dims[] = {__VA_ARGS__};                                                                                       \
        int dims_num = sizeof(dims) / sizeof(dims[0]);                                                                    \
        for (int i = 1; i <= 64; ++i)                                                                                     \
        {                                                                                                                 \
            int ret = create_common_op_test_case("absval", data_type, i, layout, dims, 4, create_test_add_n_node, 0.001); \
            if (ret) { return ret; }                                                                                      \
        }                                                                                                                 \
    }

define_test_case(test_case_0, TENGINE_LAYOUT_NCHW, 1, 3, 64, 128);
define_test_case(test_case_1, TENGINE_LAYOUT_NCHW, 1, 3, 128, 128);
define_test_case(test_case_2, TENGINE_LAYOUT_NCHW, 1, 3, 128, 64);
define_test_case(test_case_3, TENGINE_LAYOUT_NCHW, 1, 3, 111, 111);
define_test_case(test_case_4, TENGINE_LAYOUT_NCHW, 1, 3, 65, 111);

#define __NHWC_SUPPORTED__ 0
#if __NHWC_SUPPORTED__
define_test_case(test_case_5, TENGINE_LAYOUT_NHWC, 1, 64, 128, 3);
define_test_case(test_case_6, TENGINE_LAYOUT_NHWC, 1, 128, 128, 3);
define_test_case(test_case_7, TENGINE_LAYOUT_NHWC, 1, 128, 64, 3);
define_test_case(test_case_8, TENGINE_LAYOUT_NHWC, 1, 111, 111, 3);
define_test_case(test_case_9, TENGINE_LAYOUT_NHWC, 1, 65, 111, 3);
#endif

int main(void)
{
    return test_case_0() || test_case_1() || test_case_2() || test_case_3() || test_case_4()
#if __NHWC_SUPPORTED__
           || test_case_5() || test_case_6() || test_case_7() || test_case_8() || test_case_9()
#endif
        ;
}
