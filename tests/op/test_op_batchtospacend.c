#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "operator/prototype/batchtospacend_param.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include "util/vector.h"

static int __min(const int n, const int m)
{
    return n < m ? n : m;
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

static int op_test_case(const int crop_left, const int crop_right, const int crop_bottom, const int crop_top, const int dilation_x, const int dilation_y)
{
    struct batchtospacend_param params = {
        .crop_top = crop_top,
        .crop_bottom = crop_bottom,
        .crop_left = crop_left,
        .crop_right = crop_right,
        .dilation_x = dilation_x,
        .dilation_y = dilation_y};

    int dims[4] = {rand_int(1, 10) * params.dilation_x * params.dilation_y, rand_int(1, 128), rand_int(1, 128), rand_int(1, 128)};

    const int expand = dims[0] / (params.dilation_x * params.dilation_y);

    int h = expand * dims[2];
    int w = expand * dims[3];

    if (params.crop_right > h)
    {
        dims[2] = params.crop_right / expand + 1;
    }

    if (params.crop_bottom > w)
    {
        dims[3] = params.crop_bottom / expand + 1;
    }

    struct data_buffer* input = create_data_buffer(dims, 4, TENGINE_DT_FP32);
    vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
    push_vector_data(inputs, &input);

    int ret = create_common_op_test_case(OP_BATCHTOSPACEND_NAME, &params, sizeof(params), inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
    if (ret)
    {
        fprintf(stderr, "test op batchtospacend failed.");
        return ret;
    }

    return 0;
}

int main(void)
{
    return op_test_case(0, 0, 0, 0, 1, 1) || op_test_case(1, 2, 1, 2, 1, 2) || op_test_case(1, 1, 1, 1, 2, 2);
}
