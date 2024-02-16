#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/vector.h"

static int test_op_case()
{
    // broadmul 只支持一个维度的广播，例如[2, 2, 3] * [2, 2, 1]是支持的, 但是[2, 2, 3] * [2, 1, 1]不支持
    // broadmul 只支持input1向input0广播，例如[2, 2, 3] * [2, 2, 1]是支持的 但是[2, 2, 1] * [2, 2, 3]是不支持的, 当然 [2, 1, 2] * [1, 2, 1]也是不支持的
    // broadmul 要求input0 input1最后一维必须相等
    for (int loop = 0; loop < 10; ++loop)
    {
        int dims1[4] = {rand_int(10, 64), rand_int(10, 64), rand_int(10, 64), rand_int(10, 64)};

        int i = rand() % 3;
        int dims2[4] = {0};

        memcpy(dims2, dims1, sizeof(dims1));
        dims2[i] = 1;

        struct data_buffer* input1 = create_data_buffer(dims1, 4, TENGINE_DT_FP32);
        struct data_buffer* input2 = create_data_buffer(dims2, 4, TENGINE_DT_FP32);
        vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);

        push_vector_data(inputs, &input1);
        push_vector_data(inputs, &input2);

        int ret = create_common_op_test_case(OP_BROADMUL_NAME, NULL, 0, inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
        if (ret)
        {
            fprintf(stderr, "test op %s failed. ret = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", OP_BROADMUL_NAME, ret, dims1[0], dims1[1], dims1[2], dims1[3], dims2[0], dims2[1], dims2[2], dims2[3]);
            return ret;
        }
        else
        {
            fprintf(stderr, "test op %s pass. ret = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", OP_BROADMUL_NAME, ret, dims1[0], dims1[1], dims1[2], dims1[3], dims2[0], dims2[1], dims2[2], dims2[3]);
        }

        release_vector(inputs);
    }
}

int main(void)
{
    time_t tim = time(NULL);
    srand((unsigned int)tim);
    return test_op_case();
}
