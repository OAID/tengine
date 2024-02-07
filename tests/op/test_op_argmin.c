#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "operator/prototype/argmax_param.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"

#define define_common_test_case(__op_name, __case_name, __layout, __axis, __keepdims, ...)                           \
    static int __case_name()                                                                                         \
    {                                                                                                                \
        int data_type = TENGINE_DT_FP32;                                                                             \
        int layout = __layout;                                                                                       \
        int dims[] = {__VA_ARGS__};                                                                                  \
        int dims_num = sizeof(dims) / sizeof(dims[0]);                                                               \
        argmax_param_t param = {.axis = __axis, .keepdims = __keepdims};                                             \
        vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);                   \
        struct data_buffer* input = create_data_buffer_fp32(dims, sizeof(dims) / sizeof(int));                       \
        push_vector_data(inputs, &input);                                                                            \
        int ret = create_common_op_test_case(__op_name, &param, sizeof(param), inputs, 1, data_type, layout, 0.001); \
        if (ret) return ret;                                                                                         \
        release_vector(inputs);                                                                                      \
        fprintf(stderr, "test case pass, axis=%d, keepdims: %d\n", __axis, __keepdims);                              \
        return 0;                                                                                                    \
    }

#define define_test_case(__case_name, __layout, ...)                                        \
    define_common_test_case(OP_ARGMIN_NAME, __case_name##_00, __layout, 0, 0, __VA_ARGS__); \
    define_common_test_case(OP_ARGMIN_NAME, __case_name##_01, __layout, 1, 0, __VA_ARGS__); \
    define_common_test_case(OP_ARGMIN_NAME, __case_name##_02, __layout, 2, 0, __VA_ARGS__); \
    define_common_test_case(OP_ARGMIN_NAME, __case_name##_10, __layout, 0, 1, __VA_ARGS__); \
    define_common_test_case(OP_ARGMIN_NAME, __case_name##_11, __layout, 1, 1, __VA_ARGS__); \
    define_common_test_case(OP_ARGMIN_NAME, __case_name##_12, __layout, 2, 1, __VA_ARGS__); \
    static int __case_name()                                                                \
    {                                                                                       \
        __case_name##_00();                                                                 \
        __case_name##_01();                                                                 \
        __case_name##_02();                                                                 \
        __case_name##_10();                                                                 \
        __case_name##_11();                                                                 \
        __case_name##_12();                                                                 \
    }

define_test_case(op_test_case_0, TENGINE_LAYOUT_NCHW, 3, 64, 128);
define_test_case(op_test_case_1, TENGINE_LAYOUT_NCHW, 3, 128, 128);
define_test_case(op_test_case_2, TENGINE_LAYOUT_NCHW, 3, 128, 64);
define_test_case(op_test_case_3, TENGINE_LAYOUT_NCHW, 3, 111, 111);
define_test_case(op_test_case_4, TENGINE_LAYOUT_NCHW, 3, 65, 111);

#define __NHWC_SUPPORTED__ 0
#if __NHWC_SUPPORTED__
#endif

int main(void)
{
    return op_test_case_0() || op_test_case_1() || op_test_case_2() || op_test_case_3() || op_test_case_4();
}
