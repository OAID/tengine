#include "api/c_api.h"
#include "graph/graph.h"
#include "graph/node.h"
#include "test_op.h"
#include "tengine/c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util/vector.h"
#include "operator/prototype/convolution_param.h"

static int max(int lhs, int rhs)
{
    return lhs > rhs ? lhs : rhs;
}

static int test_conv_op_case(int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w)
{
    const int real_h = (kernel_h - 1) * dilation_h + stride_h + 1;
    const int real_w = (kernel_w - 1) * dilation_w + stride_w + 1;

    const int max_h = max(real_h + 1, 32);
    const int max_w = max(real_w + 1, 32);

    for (int i = 0; i < 10; ++i)
    {
        int dims[4] = {rand_int(2, 8), rand_int(2, 12), rand_int(real_h, max_h), rand_int(real_w, max_w)};
        int kernel_shape[] = {rand_int(2, 32), dims[1], kernel_h, kernel_w};

        vector_t* inputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);

        struct data_buffer* input = create_data_buffer(dims, 4, TENGINE_DT_FP32);
        struct data_buffer* filter = create_data_buffer(kernel_shape, 4, TENGINE_DT_FP32);
        push_vector_data(inputs, &input);
        push_vector_data(inputs, &filter);

        struct conv_param params = {.kernel_h = kernel_shape[2], .kernel_w = kernel_shape[3], .stride_h = stride_h, .stride_w = stride_w, .pad_h0 = pad_h, .pad_h1 = pad_h, .pad_w0 = pad_w, .pad_w1 = pad_w, .dilation_h = dilation_h, .dilation_w = dilation_w, .input_channel = kernel_shape[1], .output_channel = kernel_shape[0], .group = 1, .activation = -1, .wino_off = 1};

        int ret = create_common_op_test_case(OP_CONV_NAME, &params, sizeof(params), inputs, 1, TENGINE_DT_FP32, TENGINE_LAYOUT_NCHW, 0.001);
        release_vector(inputs);

        if (ret)
        {
            fprintf(stderr, "test conv op failed: %d, kernel_h = %d, kernel_w = %d, pad_h = %d, pad_w = %d, stride_h = %d, stride_w = %d, dilation_h = %d, dilation_w = %d, input dims = {%d, %d, %d, %d}, kernel dims = {%d, %d, %d, %d}\n", ret, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w, dims[0], dims[1], dims[2], dims[3], kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3]);
            return ret;
        }
    }

    fprintf(stderr, "test conv op pass, kernel_h = %d, kernel_w = %d, pad_h = %d, pad_w = %d, stride_h = %d, stride_w = %d, dilation_h = %d, dilation_w = %d\n", kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w);
    return 0;
}

#define __define_test_conv_op(kh, kw)                          \
    static int test_conv_op_##kh##x##kw()                      \
    {                                                          \
        return test_conv_op_case(kh, kw, 0, 0, 1, 1, 1, 1)     \
               || test_conv_op_case(kh, kw, 1, 1, 1, 1, 1, 1)  \
               || test_conv_op_case(kh, kw, 2, 2, 1, 1, 1, 1)  \
               || test_conv_op_case(kh, kw, 3, 3, 1, 1, 1, 1)  \
               || test_conv_op_case(kh, kw, 3, 1, 1, 1, 1, 1)  \
               || test_conv_op_case(kh, kw, 1, 3, 1, 1, 1, 1)  \
               || test_conv_op_case(kh, kw, 1, 3, 2, 2, 1, 1)  \
               || test_conv_op_case(kh, kw, 1, 3, 3, 3, 1, 1)  \
               || test_conv_op_case(kh, kw, 1, 3, 3, 1, 1, 1)  \
               || test_conv_op_case(kh, kw, 1, 3, 1, 3, 1, 1)  \
               || test_conv_op_case(kh, kw, 1, 3, 1, 3, 2, 2)  \
               || test_conv_op_case(kh, kw, 1, 3, 1, 3, 3, 3)  \
               || test_conv_op_case(kh, kw, 1, 3, 1, 3, 1, 3)  \
               || test_conv_op_case(kh, kw, 1, 3, 1, 3, 3, 1); \
    }

__define_test_conv_op(3, 3);
__define_test_conv_op(1, 1);

int main(void)
{
    time_t tim = time(NULL);
    srand((unsigned int)tim);
    return test_conv_op_1x1() || test_conv_op_3x3();
}
