#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "operator/prototype/batchnorm_param.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"

static void allocate_bn_inputs(vector_t* inputs, const int* dims, const int dim_num)
{
    struct data_buffer* input = create_data_buffer_fp32(dims, dim_num);
    struct data_buffer *mean, *var, *gamma, *beta;

    int dim = dims[1];
    mean = create_data_buffer_fp32(&dim, 1);
    var = create_data_buffer_fp32(&dim, 1);
    gamma = create_data_buffer_fp32(&dim, 1);
    beta = create_data_buffer_fp32(&dim, 1);

    push_vector_data(inputs, &input);
    push_vector_data(inputs, &gamma);
    push_vector_data(inputs, &beta);
    push_vector_data(inputs, &mean);
    push_vector_data(inputs, &var);
}

static int __max(const int n, const int m)
{
    return n > m ? n : m;
}

static void shuffle_array(int* arr, const int n)
{
    for (int i = 0; i < 20 * n; ++i)
    {
        int a = rand() % n;
        int b = rand() % n;
        int bak = arr[a];
        arr[a] = arr[b];
        arr[b] = bak;
    }
}

int op_test_case_0()
{
    int dims[4];
    for (int i = 0; i < 10; ++i)
    {
#define __run_test_case(__dim_num, __caffe_flavor)                                                                                              \
    do {                                                                                                                                        \
        dims[0] = __max(rand() % 10, 1);                                                                                                        \
        dims[1] = __max(rand() % 128, 1);                                                                                                       \
        dims[2] = __max(rand() % 128, 1);                                                                                                       \
        dims[3] = __max(rand() % 128, 1);                                                                                                       \
        shuffle_array(dims, 4);                                                                                                                 \
        float rescale_factor = random_float(-100.0f, 100.0f);                                                                                   \
        rescale_factor = rand() % 100 > 50 ? rescale_factor : .0;                                                                               \
        batchnorm_param_t param = {.caffe_flavor = __caffe_flavor, .rescale_factor = rescale_factor, .eps = 0.001};                             \
        vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);                                              \
        allocate_bn_inputs(inputs, dims, __dim_num);                                                                                            \
        int ret = create_common_op_test_case(OP_BATCHNORM_NAME, &param, sizeof(param), inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001); \
        release_vector(inputs);                                                                                                                 \
        if (ret) return ret;                                                                                                                    \
        fprintf(stderr, "batchnorm op test pass: dim_num = %d, caffe_flavor = %d\n", __dim_num, __caffe_flavor);                                \
    } while (0)

        __run_test_case(2, 0);
        __run_test_case(3, 0);
        __run_test_case(4, 0);
        __run_test_case(2, 1);
        __run_test_case(3, 1);
        __run_test_case(4, 1);
    }
}

int main(void)
{
    time_t tim = time(NULL);
    srand((unsigned int)tim);
    return op_test_case_0();
}
