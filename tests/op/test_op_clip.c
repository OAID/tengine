#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "tengine/c_api.h"
#include "operator/prototype/clip_param.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"

static int test_ceil_op()
{
    for (int i = 0; i < 10; ++i)
    {
        struct clip_param params = {.min = random_float(-1.0, 0.0), .max = random_float(0.0, 1.0)};
        int dims[4] = {rand_int(10, 64), rand_int(10, 64), rand_int(10, 64), rand_int(10, 64)};
        struct data_buffer* input = create_data_buffer(dims, 4, TENGINE_DT_FP32);
        vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
        push_vector_data(inputs, &input);

        int ret = create_common_op_test_case(OP_CLIP_NAME, &params, sizeof(params), inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
        if (ret)
        {
            return ret;
        }

        release_vector(inputs);

        input = create_data_buffer(dims, 4, TENGINE_DT_UINT8);
        inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
        push_vector_data(inputs, &input);

        ret = create_common_op_test_case(OP_CLIP_NAME, &params, sizeof(params), inputs, 1, TENGINE_DT_UINT8, TENGINE_LAYOUT_NCHW, 0.001);

        if (ret) { return ret; }

        release_vector(inputs);

        input = create_data_buffer(dims, 4, TENGINE_DT_INT8);
        inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
        push_vector_data(inputs, &input);

        ret = create_common_op_test_case(OP_CLIP_NAME, &params, sizeof(params), inputs, 1, TENGINE_DT_INT8, TENGINE_LAYOUT_NCHW, 0.001);

        if (ret) { return ret; }

        release_vector(inputs);
    }
    return 0;
}

int main(void)
{
    time_t tim = time(NULL);
    srand((unsigned int)tim);
    return test_ceil_op();
}
