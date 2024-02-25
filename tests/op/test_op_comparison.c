#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/vector.h"
#include "operator/prototype/comparison_param.h"

static int get_total_size(const int* dims, const int n)
{
    int s = 1;
    for (int i = 0; i < n; ++i)
    {
        s *= dims[i];
    }
    return s;
}

static void random_mask(float* data, const int size)
{
    int n = (int)(0.5f * size);
    for (int i = 0; i < n; ++i)
    {
        int k = rand() % n;
        data[k] = random_float(-1.2f, 1.2f);
    }
}

static int do_comparison_test(const int* dims1, const int* dims2, const int n1, const int n2)
{
    for (int i = 0; i <= 5; ++i)
    {
        struct comparison_param params = {.type = i};

        struct data_buffer* input = create_data_buffer(dims1, n1, TENGINE_DT_FP32);
        struct data_buffer* input1 = create_data_buffer(dims2, n2, TENGINE_DT_FP32);
        vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
        push_vector_data(inputs, &input);
        push_vector_data(inputs, &input1);

        int ret = create_common_op_test_case(OP_COMPARISON_NAME, &params, sizeof(params), inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
        if (ret)
        {
            fprintf(stderr, "test comparison op failed: %d, type = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", ret, i, dims1[0], dims1[1], dims1[2], dims1[3], dims2[0], dims2[1], dims2[2], dims2[3]);
            release_vector(inputs);
            return ret;
        }

        const int total_size1 = get_total_size(dims1, n1);
        const int total_size2 = get_total_size(dims2, n2);
        if (total_size1 > total_size2)
        {
            random_mask(input->data, total_size1);
        }
        else
        {
            random_mask(input1->data, total_size2);
        }

        ret = create_common_op_test_case(OP_COMPARISON_NAME, &params, sizeof(params), inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
        release_vector(inputs);
        if (ret)
        {
            fprintf(stderr, "test comparison op after masked failed: %d, type = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", ret, i, dims1[0], dims1[1], dims1[2], dims1[3], dims2[0], dims2[1], dims2[2], dims2[3]);
            return ret;
        }
    }

    fprintf(stderr, "test comparison op pass\n");
    return 0;
}

static int test_comparison_op()
{
    int dims1[] = {rand_int(2, 10), rand_int(10, 32), rand_int(10, 32), rand_int(10, 32)};
    int dims2[4] = {0};

    memcpy(dims2, dims1, sizeof(dims1));
    int ret = do_comparison_test(dims1, dims2, 4, 4);
    if (ret) { return ret; }

    dims2[0] = 1;
    ret = do_comparison_test(dims1, dims2, 4, 1) || do_comparison_test(dims2, dims1, 1, 4);
    if (ret) return ret;

    dims2[0] = dims1[1];

    return do_comparison_test(dims1, dims2, 4, 1) || do_comparison_test(dims2, dims1, 1, 4);
}

int main(void)
{
    time_t tim = time(NULL);
    srand((unsigned int)tim);
    return test_comparison_op();
}
