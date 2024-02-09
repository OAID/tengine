#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"

#define define_common_test_case(__op_name, __case_name, __layout, ...)                                                                              \
    static int __case_name()                                                                                                                        \
    {                                                                                                                                               \
        int data_type = TENGINE_DT_FP32;                                                                                                            \
        int layout = __layout;                                                                                                                      \
        int dims[] = {__VA_ARGS__};                                                                                                                 \
        int dims_num = sizeof(dims) / sizeof(dims[0]);                                                                                              \
        vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);                                                  \
        struct data_buffer* input = create_data_buffer(dims, dims_num, data_type);                                                                  \
        push_vector_data(inputs, &input);                                                                                                           \
        struct data_buffer* bias = create_data_buffer(&dims[1], 1, data_type);                                                                      \
        push_vector_data(inputs, &bias);                                                                                                            \
        int ret = create_common_op_test_case(__op_name, NULL, 0, inputs, 1, data_type, layout, 0.001);                                              \
        if (ret) { fprintf(stderr, "test op %s failed: ret = %d, dims = {%d, %d, %d, %d}\n", __op_name, ret, dims[0], dims[1], dims[2], dims[3]); } \
        release_vector(inputs);                                                                                                                     \
        return 0;                                                                                                                                   \
    }

#define define_test_case(__case_name, __layout, ...) define_common_test_case(OP_BIAS_NAME, __case_name, __layout, __VA_ARGS__)

define_test_case(op_test_case_0, TENGINE_LAYOUT_NCHW, 1, 3, 64, 128);
define_test_case(op_test_case_1, TENGINE_LAYOUT_NCHW, 1, 3, 128, 128);
define_test_case(op_test_case_2, TENGINE_LAYOUT_NCHW, 1, 3, 128, 64);
define_test_case(op_test_case_3, TENGINE_LAYOUT_NCHW, 1, 3, 111, 111);
define_test_case(op_test_case_4, TENGINE_LAYOUT_NCHW, 1, 3, 65, 111);

int main(void)
{
    return op_test_case_0() || op_test_case_1() || op_test_case_2() || op_test_case_3() || op_test_case_4();
}
