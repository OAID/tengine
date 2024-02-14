#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "operator/prototype/cast_param.h"
#include "test_op.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"

static int test_cast_op(const int from, const int to)
{
    int dims[4] = {rand_int(10, 64), rand_int(10, 64), rand_int(10, 64), rand_int(10, 64)};
    struct data_buffer* input = create_data_buffer(dims, 4, from);
    vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
    push_vector_data(inputs, &input);

    struct cast_param params = {.type_from = from, .type_to = to};

    int ret = create_common_op_test_case(OP_CAST_NAME, &params, sizeof(params), inputs, 1, to, TENGINE_LAYOUT_NCHW, 0.001);
    if (ret)
    {
        fprintf(stderr, "test op %s failed. ret = %d, dims1 = {%d, %d, %d, %d}, from type = %d, to type = %d\n", OP_CAST_NAME, ret, dims[0], dims[1], dims[2], dims[3], from, to);
        return ret;
    }

    release_vector(inputs);
    return 0;
}

int main(void)
{
    time_t tim = time(NULL);
    srand((unsigned int)tim);
    return test_cast_op(TENGINE_DT_FP32, TENGINE_DT_FP16)
           || test_cast_op(TENGINE_DT_FP16, TENGINE_DT_FP32)
           || test_cast_op(TENGINE_DT_FP32, TENGINE_DT_UINT8)
           || test_cast_op(TENGINE_DT_UINT8, TENGINE_DT_FP32)
           || test_cast_op(TENGINE_DT_FP32, TENGINE_DT_FP32)
           || test_cast_op(TENGINE_DT_UINT8, TENGINE_DT_UINT8);
}
