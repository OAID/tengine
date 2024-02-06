#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"

#define define_test_case(__func, __layout, ...)                                                             \
    static int __func()                                                                                     \
    {                                                                                                       \
        int data_type = TENGINE_DT_FP32;                                                                    \
        int layout = __layout;                                                                              \
        int dims[] = {__VA_ARGS__};                                                                         \
        int dims_num = sizeof(dims) / sizeof(dims[0]);                                                      \
        vector_t* inputs = create_vector(sizeof(struct data_buffer), free_data_buffer_in_vector);           \
        struct data_buffer* input = create_data_buffer_fp32(dims, sizeof(dims) / sizeof(int));              \
        push_vector_data(inputs, &input);                                                                   \
        int ret = create_common_op_test_case(OP_ABSVAL_NAME, NULL, 0, inputs, 1, data_type, layout, 0.001); \
        release_vector(inputs);                                                                             \
        return ret;                                                                                         \
    }

define_test_case(absval_op_test_case_0, TENGINE_LAYOUT_NCHW, 1, 3, 64, 128);
define_test_case(absval_op_test_case_1, TENGINE_LAYOUT_NCHW, 1, 3, 128, 128);
define_test_case(absval_op_test_case_2, TENGINE_LAYOUT_NCHW, 1, 3, 128, 64);
define_test_case(absval_op_test_case_3, TENGINE_LAYOUT_NCHW, 1, 3, 111, 111);
define_test_case(absval_op_test_case_4, TENGINE_LAYOUT_NCHW, 1, 3, 65, 111);

#define __NHWC_SUPPORTED__ 0
#if __NHWC_SUPPORTED__
#endif

int main(void)
{
    return absval_op_test_case_0() || absval_op_test_case_1() || absval_op_test_case_2() || absval_op_test_case_3() || absval_op_test_case_4()
#if __NHWC_SUPPORTED__
#endif
        ;
}
