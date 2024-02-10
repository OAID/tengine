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
    int dims1[4] = {rand_int(1, 128), rand_int(1, 128), rand_int(1, 128), rand_int(1, 128)};
    int i = rand() % 4;
    int dims2[4] = {0};

    memcpy(dims2, dims1, sizeof(dims1));
    dims1[i] = 1;
    dims2[i] = rand_int(1, 32);

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

    input1 = create_data_buffer(dims1, 4, TENGINE_DT_FP32);
    input2 = create_data_buffer(dims2, 4, TENGINE_DT_FP32);
    set_vector_data(inputs, 0, &input2);
    set_vector_data(inputs, 1, &input1);

    ret = create_common_op_test_case(OP_BROADMUL_NAME, NULL, 0, inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
    if (ret)
    {
        fprintf(stderr, "test op %s failed. ret = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", OP_BROADMUL_NAME, ret, dims2[0], dims2[1], dims2[2], dims2[3], dims1[0], dims1[1], dims1[2], dims1[3]);
        return ret;
    }

    release_vector(inputs);

    int k = i;
    for (;;)
    {
        k = rand() % 4;
        if (k != i)
        {
            break;
        }
    }

    dims1[k] = 1;
    dims2[i] = rand_int(1, 32);

    inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
    input1 = create_data_buffer(dims1, 4, TENGINE_DT_FP32);
    input2 = create_data_buffer(dims2, 4, TENGINE_DT_FP32);
    push_vector_data(inputs, &input1);
    push_vector_data(inputs, &input2);

    ret = create_common_op_test_case(OP_BROADMUL_NAME, NULL, 0, inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
    if (ret)
    {
        fprintf(stderr, "test op %s failed. ret = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", OP_BROADMUL_NAME, ret, dims1[0], dims1[1], dims1[2], dims1[3], dims2[0], dims2[1], dims2[2], dims2[3]);
        return ret;
    }

    input1 = create_data_buffer(dims1, 4, TENGINE_DT_FP32);
    input2 = create_data_buffer(dims2, 4, TENGINE_DT_FP32);
    set_vector_data(inputs, 0, &input2);
    set_vector_data(inputs, 1, &input1);

    ret = create_common_op_test_case(OP_BROADMUL_NAME, NULL, 0, inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
    if (ret)
    {
        fprintf(stderr, "test op %s failed. ret = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", OP_BROADMUL_NAME, ret, dims2[0], dims2[1], dims2[2], dims2[3], dims1[0], dims1[1], dims1[2], dims1[3]);
        return ret;
    }

    release_vector(inputs);

    int j = i;
    for (;;)
    {
        j = rand() % 4;
        if (j != i && j != k)
        {
            break;
        }
    }

    dims1[j] = 1;
    dims2[j] = rand_int(1, 32);

    inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
    input1 = create_data_buffer(dims1, 4, TENGINE_DT_FP32);
    input2 = create_data_buffer(dims1, 4, TENGINE_DT_FP32);
    push_vector_data(inputs, &input1);
    push_vector_data(inputs, &input2);

    ret = create_common_op_test_case(OP_BROADMUL_NAME, NULL, 0, inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
    if (ret)
    {
        fprintf(stderr, "test op %s failed. ret = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", OP_BROADMUL_NAME, ret, dims1[0], dims1[1], dims1[2], dims1[3], dims2[0], dims2[1], dims2[2], dims2[3]);
        return ret;
    }

    input1 = create_data_buffer(dims1, 4, TENGINE_DT_FP32);
    input2 = create_data_buffer(dims2, 4, TENGINE_DT_FP32);
    set_vector_data(inputs, 0, &input2);
    set_vector_data(inputs, 1, &input1);

    ret = create_common_op_test_case(OP_BROADMUL_NAME, NULL, 0, inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
    if (ret)
    {
        fprintf(stderr, "test op %s failed. ret = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", OP_BROADMUL_NAME, ret, dims2[0], dims2[1], dims2[2], dims2[3], dims1[0], dims1[1], dims1[2], dims1[3]);
        return ret;
    }

    release_vector(inputs);
}

int main(void)
{
    return test_op_case();
}
