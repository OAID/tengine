#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"
#include "operator/prototype/comparison_param.h"

static int do_comparison_test(vector_t* inputs, int type)
{
    struct comparison_param params = {.type = type};
    return create_common_op_test_case(OP_COMPARISON_NAME, &params, sizeof(params), inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
}

static int test_comparison_op()
{
    for (int i = 0; i <= 5; ++i)
    {
        int dims[4] = {rand_int(10, 64), rand_int(10, 64), rand_int(10, 64), rand_int(10, 64)};
        struct data_buffer* input = create_data_buffer(dims, 4, TENGINE_DT_FP32);
        struct data_buffer* input1 = create_data_buffer(dims, 4, TENGINE_DT_FP32);
        vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
        push_vector_data(inputs, &input);
        push_vector_data(inputs, &input1);

        int ret = do_comparison_test(inputs, i) || do_comparison_test(inputs, i) || do_comparison_test(inputs, i);
        if (ret)
        {
            return ret;
        }

        int n = (int)(dims[0] * dims[1] * dims[2] * dims[3] * 0.5);
        float* p1 = input->data;
        float* p2 = input1->data;
        for (int i = 0; i < n; ++i)
        {
            int k = rand() % n;
            int tmp = p1[k];
            p1[k] = p2[k];
            p2[k] = tmp;
        }

        ret = do_comparison_test(inputs, i);
        if (ret)
        {
            return ret;
        }

        release_vector(inputs);
    }
    return 0;
}

int main(void)
{
    time_t tim = time(NULL);
    srand((unsigned int)tim);
    return test_comparison_op();
}
