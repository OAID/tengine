#include "test_op.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"

int create_test_absval_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    node_t test_node = create_graph_node(graph, node_name, OP_ABSVAL_NAME);
    if (NULL == test_node)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    set_node_input_tensor(test_node, 0, input_tensor);

    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    if (!output_tensor)
    {
        fprintf(stderr, "create graph output tensor failed.\n");
        return -1;
    }

    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);
    return 0;
}

#define define_absval_test_case(__func, __layout, ...)                                                           \
    int __func()                                                                                                 \
    {                                                                                                            \
        const char* test_node_name = "absval";                                                                   \
        int data_type = TENGINE_DT_FP32;                                                                         \
        int layout = __layout;                                                                                   \
        int dims[] = {__VA_ARGS__};                                                                              \
        int dims_num = sizeof(dims) / sizeof(dims[0]);                                                           \
        return create_common_op_test_case("absval", data_type, layout, dims, 4, create_test_absval_node, 0.001); \
    }

define_absval_test_case(absval_op_test_case_0, TENGINE_LAYOUT_NCHW, 1, 3, 64, 128);
define_absval_test_case(absval_op_test_case_1, TENGINE_LAYOUT_NCHW, 1, 3, 128, 128);
define_absval_test_case(absval_op_test_case_2, TENGINE_LAYOUT_NCHW, 1, 3, 128, 64);
define_absval_test_case(absval_op_test_case_3, TENGINE_LAYOUT_NCHW, 1, 3, 111, 111);
define_absval_test_case(absval_op_test_case_4, TENGINE_LAYOUT_NCHW, 1, 3, 65, 111);

#define __NHWC_SUPPORTED__ 0
#if __NHWC_SUPPORTED__
define_absval_test_case(absval_op_test_case_5, TENGINE_LAYOUT_NHWC, 1, 64, 128, 3);
define_absval_test_case(absval_op_test_case_6, TENGINE_LAYOUT_NHWC, 1, 128, 128, 3);
define_absval_test_case(absval_op_test_case_7, TENGINE_LAYOUT_NHWC, 1, 128, 64, 3);
define_absval_test_case(absval_op_test_case_8, TENGINE_LAYOUT_NHWC, 1, 111, 111, 3);
define_absval_test_case(absval_op_test_case_9, TENGINE_LAYOUT_NHWC, 1, 65, 111, 3);
#endif

int main(void)
{
    return absval_op_test_case_0() || absval_op_test_case_1() || absval_op_test_case_2() || absval_op_test_case_3() || absval_op_test_case_4()
#if __NHWC_SUPPORTED__
           || absval_op_test_case_5() || absval_op_test_case_6() || absval_op_test_case_7() || absval_op_test_case_8() || absval_op_test_case_9()
#endif
        ;
}
