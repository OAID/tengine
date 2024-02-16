#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "convolution_param.h"
#include "graph/tensor.h"
#include "op/conv/x86/conv_kernel_x86.h"
#include "utility/sys_port.h"
#include <errno.h>
#include <string.h>

#define PER_OUT_CHAN 8
#define min(a, b)    ((a) < (b) ? (a) : (b))

extern void sgemm_8x8_rv64(const float* cur_col, const float* cur_kernel, const float* bias, const int act, float* cur_output, const int output_xy, const int kernel_size, const int n);
extern void im2col_tile8(float* input, float* col, int in_c, int in_w, int in_h, int k_w, int k_h, int s_w, int s_h, int d_w,
                         int d_h, int pad_w0, int pad_w1, int pad_h0, int pad_h1, int out_w, int out_h, int num_thread);

static void interleave_kernel(float* kernel, float* kernel_interleaved, int kernel_chan, int kernel_size)
{
    int i, j, k;
    float* cur_kernel[PER_OUT_CHAN];
    float* cur_kernel_interleaved = kernel_interleaved;

    // interleave PER_OUT_CHAN kernels
    for (i = 0; i + PER_OUT_CHAN - 1 < kernel_chan; i += PER_OUT_CHAN)
    {
        for (k = 0; k < PER_OUT_CHAN; k++)
            cur_kernel[k] = kernel + kernel_size * (i + k);
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < PER_OUT_CHAN; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
        }
    }

    // last 7 kernel
    for (k = 0; i + k < kernel_chan; k++)
        cur_kernel[k] = kernel + kernel_size * (i + k);

    if ((kernel_chan & 0x7) == 7)
    {
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < 7; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
    else if ((kernel_chan & 0x7) == 6)
    {
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < 6; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
    else if ((kernel_chan & 0x7) == 5)
    {
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < 5; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
    else if ((kernel_chan & 0x7) == 4)
    {
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < 4; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
    else if ((kernel_chan & 0x7) == 3)
    {
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < 3; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
    else if ((kernel_chan & 0x7) == 2)
    {
        for (j = 0; j < kernel_size; j++)
        {
            for (k = 0; k < 2; k++)
                *(cur_kernel_interleaved++) = cur_kernel[k][j];
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
    else if ((kernel_chan & 0x7) == 1)
    {
        for (j = 0; j < kernel_size; j++)
        {
            *(cur_kernel_interleaved++) = cur_kernel[0][j];
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
            *(cur_kernel_interleaved++) = 0.f;
        }
    }
}

/* kernel interleave */
static void interleave(struct tensor* filter, struct conv_priv_info* priv_info, struct conv_param* param)
{
    int group = param->group;
    int in_c = filter->dims[1];
    int kernel_h = filter->dims[2];
    int kernel_w = filter->dims[3];
    int kernel_size = in_c * kernel_h * kernel_w;

    int out_chan = filter->dims[0] / group;
    int out_chan_align8 = (out_chan + 7) / 8 * 8;

    int kernel_size_algin = kernel_size * out_chan_align8;
    int kernel_size_group = kernel_size * out_chan;

    float* kernel = filter->data;

    float* interleave_buf = priv_info->interleave_buffer;
    for (int g = 0; g < group; g++)
    {
        float* cur_kernel = kernel + g * kernel_size_group;
        float* cur_interleave = interleave_buf + g * kernel_size_algin;
        interleave_kernel(cur_kernel, cur_interleave, out_chan, kernel_size);
    }
}

int conv_hcl_get_shared_mem_size_rv64_tile8(struct tensor* input_tensor, struct tensor* output_tensor, struct conv_param* param)
{
    int kernel_size = param->kernel_h * param->kernel_w * param->input_channel / param->group;
    int cstep = output_tensor->dims[2] * output_tensor->dims[3];

    cstep = (cstep + 7) / 8 * 8; //align to 8
    int mem_size = input_tensor->elem_size * cstep * kernel_size + 128 * sizeof(float);
    return mem_size;
}

int conv_hcl_prerun_tile8(struct node* ir_node, struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* output_tensor, struct conv_priv_info* info, struct conv_param* param)
{
    // alloc im2col buffer = kernel_size * out_xy
    if (!info->external_im2col_mem)
    {
        int mem_size = conv_hcl_get_shared_mem_size_rv64_tile8(input_tensor, output_tensor, param);
        info->im2col_buffer = sys_malloc(mem_size);
        info->im2col_buffer_size = mem_size;
    }

    // alloc kernel interleave buffer
    if (!info->external_interleave_mem)
    {
        int kernel_size = filter_tensor->dims[1] * filter_tensor->dims[2] * filter_tensor->dims[3];
        int out_chan = filter_tensor->dims[0] / param->group;
        out_chan = (out_chan + 7) / 8 * 8; //align to 8
        int mem_size = out_chan * kernel_size * filter_tensor->elem_size * param->group;
        info->interleave_buffer = sys_malloc(mem_size);
        info->interleave_buffer_size = mem_size;
    }

    // interleave kernel
    interleave(filter_tensor, info, param);
    return 0;
}

int conv_hcl_postrun_tile8(struct node* ir_node, struct conv_priv_info* info)
{
    if (!info->external_interleave_mem && info->interleave_buffer)
    {
        sys_free(info->interleave_buffer);
        info->interleave_buffer = NULL;
    }

    if (!info->external_im2col_mem && info->im2col_buffer)
    {
        sys_free(info->im2col_buffer);
        info->im2col_buffer = NULL;
    }

    return 0;
}

int conv_hcl_run_tile8(struct node* ir_node, struct tensor* input_tensor, struct tensor* filter_tensor, struct tensor* bias_tensor, struct tensor* output_tensor, struct conv_priv_info* info, struct conv_param* param, int num_thread, int cpu_affinity)
{
    int group = param->group;
    int batch = input_tensor->dims[0];
    float* input = input_tensor->data;
    float* output = output_tensor->data;
    float* bias = NULL;
    if (bias_tensor)
    {
        bias = bias_tensor->data;
    }

    int in_c = input_tensor->dims[1];
    in_c /= group;
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int input_size = in_c * in_h * in_w;

    int k_h = param->kernel_h;
    int k_w = param->kernel_w;
    int s_w = param->stride_w;
    int s_h = param->stride_h;
    int d_h = param->dilation_h;
    int d_w = param->dilation_w;
    int p_h0 = param->pad_h0;
    int p_w0 = param->pad_w0;
    int p_h1 = param->pad_h1;
    int p_w1 = param->pad_w1;
    int act = param->activation;
    int kernel_size = in_c * k_h * k_w;

    int out_c = param->output_channel / group;
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_xy = out_h * out_w;
    int output_size = out_c * out_h * out_w;
    int output_image_size = output_tensor->dims[1] * output_tensor->dims[2] * output_tensor->dims[3]; //不是8倍数怎么办

    int out_c_align8 = (out_c + 7) / 8 * 8;
    int input_image_size = in_c * in_h * in_w;
    int input_group_size = input_image_size * group;

    float* col = info->im2col_buffer; // FIXME: split by [batch, group]
    float* interleaved_kernel = info->interleave_buffer;

    for (int n = 0; n < batch; ++n)
    {
        for (int g = 0; g < group; ++g)
        {
            float* cur_input = input + n * input_image_size + g * input_size;
            //output shape: [batch, group, output_xy/8, ksize, 8]
            im2col_tile8(cur_input, col, in_c, in_w, in_h, k_w, k_h, s_w, s_h, d_w, d_h, p_w0, p_w1, p_h0, p_h1, out_w, out_h, num_thread);

            float* output_base = output + n * output_image_size + g * output_size;
            //FIXME: out_chan_ 可能不是8对齐的
            int out_chan_ = 0;
            for (; out_chan_ < out_c_align8; out_chan_ += PER_OUT_CHAN)
            {
                float* cur_kernel = interleaved_kernel + g * out_c_align8 * kernel_size + out_chan_ * kernel_size;
                float* cur_bias = bias ? bias + g * out_c + out_chan_ : NULL;
                float* cur_output = output_base + out_chan_ * out_xy;
                const int n = min(8, out_c - out_chan_);

                int col_i = 0;
                for (; col_i + 7 < out_xy; col_i += 8)
                {
                    float* cur_col = col + col_i * kernel_size;
                    sgemm_8x8_rv64(cur_col, cur_kernel, cur_bias, act, cur_output + col_i, out_xy, kernel_size, n);
                }
                if (col_i < out_xy)
                {
                    float result[64];
                    float* cur_col = (col + col_i * kernel_size);
                    sgemm_8x8_rv64(cur_col, cur_kernel, cur_bias, act, result, 8, kernel_size, n);

                    int col_end3 = (out_xy & 7);

                    for (int i = 0; i < n; i++)
                    {
                        int j = 0;
                        for (; j < (col_end3); j++)
                            *(cur_output + i * out_xy + col_i + j) = result[(i << 3) + j];
                    }
                }
            }
        }
    }

    return 0;
}
