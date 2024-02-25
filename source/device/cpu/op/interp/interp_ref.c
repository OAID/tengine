/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: qtang@openailab.com
 */

#include "interp_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>
#include <string.h>

#define INTERP_MIN(a, b) ((a) < (b) ? (a) : (b))

static void linear_coeffs(int w, int outw, int* xofs, float* alpha, int align_corner)
{
    double scale = (double)w / outw;
    if (align_corner)
    {
        scale = (double)(w - 1) / (outw - 1);
    }
    for (int dx = 0; dx < outw; dx++)
    {
        float fx = (float)((dx + 0.5) * scale - 0.5);
        if (align_corner)
        {
            fx = (float)(dx * scale);
        }

        int sx = floor(fx);
        fx -= sx;

        if (sx < 0)
        {
            sx = 0;
            fx = 0.f;
        }
        if (sx >= w - 1)
        {
            sx = w - 2;
            fx = 1.f;
        }

        xofs[dx] = sx;

        alpha[dx * 2] = 1.f - fx;
        alpha[dx * 2 + 1] = fx;
    }
}

void resize_bilinear_image(float* src, float* dst, float* alpha, int* xofs, float* beta, int* yofs, int out_h, int out_w, int in_h, int in_w)
{
    int w = out_w; //dst.w;
    int h = out_h; //dst.h;

    // loop body
    float* rowsbuf0 = (float*)sys_malloc(w * sizeof(float));
    float* rowsbuf1 = (float*)sys_malloc(w * sizeof(float));
    float* rows0 = rowsbuf0;
    float* rows1 = rowsbuf1;

    memset(rowsbuf0, 0, w * sizeof(float));
    memset(rowsbuf1, 0, w * sizeof(float));

    int prev_sy1 = -2;
    for (int dy = 0; dy < h; dy++)
    {
        int sy = yofs[dy];

        if (sy == prev_sy1)
        {
            // reuse all rows
        }
        else if (sy == prev_sy1 + 1)
        {
            // hresize one row
            float* rows0_old = rows0;
            rows0 = rows1;
            rows1 = rows0_old;
            const float* S1 = src + (sy + 1) * in_w; //src.row(sy+1);

            const float* alphap = alpha;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }
        else
        {
            // hresize two rows
            const float* S0 = src + sy * in_w;       //src.row(sy);
            const float* S1 = src + (sy + 1) * in_w; //src.row(sy+1);

            const float* alphap = alpha;
            float* rows0p = rows0;
            float* rows1p = rows1;

            for (int dx = 0; dx < w; dx++)
            {
                int sx = xofs[dx];
                const float* S0p = S0 + sx;
                const float* S1p = S1 + sx;

                float a0 = alphap[0];
                float a1 = alphap[1];
                rows0p[dx] = S0p[0] * a0 + S0p[1] * a1;
                rows1p[dx] = S1p[0] * a0 + S1p[1] * a1;

                alphap += 2;
            }
        }

        prev_sy1 = sy;

        // vresize
        float b0 = beta[0];
        float b1 = beta[1];

        float* rows0p = rows0;
        float* rows1p = rows1;
        float* Dp = dst + dy * out_w; //dst.row(dy);

        for (int dx = 0; dx < w; dx++)
        {
            float temp = *rows0p++ * b0 + *rows1p++ * b1;
            *Dp++ = temp;
        }

        beta += 2;
    }

    sys_free(rowsbuf0);
    sys_free(rowsbuf1);
}

int ref_interp_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct interp_param* param)
{
    if (param->resize_type == 1)
    {
        float* input = (float*)input_tensor->data;
        float* output = (float*)output_tensor->data;

        int batch = output_tensor->dims[0];
        int channel = output_tensor->dims[1];
        int output_h = output_tensor->dims[2];
        int output_w = output_tensor->dims[3];
        int input_h = input_tensor->dims[2];
        int input_w = input_tensor->dims[3];

        for (int n = 0; n < batch; ++n)
        {
            for (int c = 0; c < channel; c++)
            {
                for (int h = 0; h < output_h; h++)
                {
                    for (int w = 0; w < output_w; w++)
                    {
                        int in_w = w / param->width_scale;
                        int in_h = h / param->height_scale;
                        int out_idx = n * channel * output_h * output_w + c * output_h * output_w + h * output_w + w;
                        int in_idx = n * channel * input_h * input_w + c * input_w * input_h + in_h * input_w + in_w;
                        output[out_idx] = input[in_idx];
                    }
                }
            }
        }
    }
    else if (param->resize_type == 2 || param->resize_type == 4)
    {
        float* input = (float*)input_tensor->data;
        float* output = (float*)output_tensor->data;

        int batch = input_tensor->dims[0];
        int channel = input_tensor->dims[1];
        int in_h = input_tensor->dims[2];
        int in_w = input_tensor->dims[3];
        int out_h = output_tensor->dims[2];
        int out_w = output_tensor->dims[3];

        int in_channel_size = in_h * in_w;
        int out_channel_size = out_h * out_w;

        int* buf = (int*)sys_malloc((param->output_width + param->output_height + param->output_width * 2 + param->output_height * 2) * sizeof(float));

        if (buf == NULL)
        {
            TLOG_ERR("interp malloc failed!\n");
            return -1;
        }

        int* xofs = buf;                       //new int[ow];
        int* yofs = buf + param->output_width; //new int[oh];

        float* alpha = (float*)(buf + param->output_width + param->output_height);                          //new float[ow * 2];
        float* beta = (float*)(buf + param->output_width + param->output_height + param->output_width * 2); //new float[oh * 2];

        int align_corner = param->resize_type == 2 ? 0 : 1;
        linear_coeffs(in_w, out_w, xofs, alpha, align_corner);
        linear_coeffs(in_h, out_h, yofs, beta, align_corner);

        for (int q = 0; q < channel; ++q)
        {
            resize_bilinear_image(input + in_channel_size * q, output + out_channel_size * q, alpha, xofs, beta, yofs, out_h, out_w, in_h, in_w);
        }

        sys_free(buf);
    }
    else
    {
        TLOG_ERR("interp resize type %d not support!\n", param->resize_type);
        return -1;
    }

    return 0;
}

int ref_interp_int8(struct tensor* input_tensor, struct tensor* output_tensor, struct interp_param* param)
{
    /* dequant */
    int input_total_size = input_tensor->elem_num;
    int output_total_size = output_tensor->elem_num;

    int8_t* input_int8 = (int8_t*)input_tensor->data;
    int8_t* output_int8 = (int8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* input_fp32 = (float*)sys_malloc(input_total_size * sizeof(float));
    float* output_fp32 = (float*)sys_malloc(output_total_size * sizeof(float));

    for (int i = 0; i < input_total_size; i++)
    {
        input_fp32[i] = ((float)input_int8[i] - (float)input_zero) * input_scale;
    }

    /* process */
    if (param->resize_type == 1)
    {
        int batch = output_tensor->dims[0];
        int channel = output_tensor->dims[1];
        int output_h = output_tensor->dims[2];
        int output_w = output_tensor->dims[3];
        int input_h = input_tensor->dims[2];
        int input_w = input_tensor->dims[3];

        for (int n = 0; n < batch; ++n)
        {
            for (int c = 0; c < channel; c++)
            {
                for (int h = 0; h < output_h; h++)
                {
                    for (int w = 0; w < output_w; w++)
                    {
                        int in_w = w / param->width_scale;
                        int in_h = h / param->height_scale;
                        int out_idx = n * channel * output_h * output_w + c * output_h * output_w + h * output_w + w;
                        int in_idx = n * channel * input_h * input_w + c * input_w * input_h + in_h * input_w + in_w;
                        output_fp32[out_idx] = input_fp32[in_idx];
                    }
                }
            }
        }
    }
    else if (param->resize_type == 2 || param->resize_type == 4)
    {
        int batch = input_tensor->dims[0];
        int channel = input_tensor->dims[1];
        int in_h = input_tensor->dims[2];
        int in_w = input_tensor->dims[3];
        int out_h = output_tensor->dims[2];
        int out_w = output_tensor->dims[3];

        int in_channel_size = in_h * in_w;
        int out_channel_size = out_h * out_w;

        int* buf = (int*)sys_malloc((param->output_width + param->output_height + param->output_width * 2 + param->output_height * 2) * sizeof(float));

        if (buf == NULL)
        {
            TLOG_ERR("interp malloc failed!\n");
            return -1;
        }

        int* xofs = buf;                       //new int[ow];
        int* yofs = buf + param->output_width; //new int[oh];

        float* alpha = (float*)(buf + param->output_width + param->output_height);                          //new float[ow * 2];
        float* beta = (float*)(buf + param->output_width + param->output_height + param->output_width * 2); //new float[oh * 2];

        int align_corner = param->resize_type == 2 ? 0 : 1;
        linear_coeffs(in_w, out_w, xofs, alpha, align_corner);
        linear_coeffs(in_h, out_h, yofs, beta, align_corner);

        for (int q = 0; q < channel; ++q)
        {
            resize_bilinear_image(input_fp32 + in_channel_size * q, output_fp32 + out_channel_size * q, alpha, xofs, beta, yofs, out_h, out_w, in_h, in_w);
        }

        sys_free(buf);
    }
    else
    {
        TLOG_ERR("interp resize type %d not support!\n", param->resize_type);
        return -1;
    }

    /* quant */
    for (int i = 0; i < output_total_size; i++)
    {
        int idata = round(output_fp32[i] / output_scale + output_zero);
        if (idata > 127)
            idata = 127;
        else if (idata < -127)
            idata = -127;
        output_int8[i] = idata;
    }

    sys_free(input_fp32);
    sys_free(output_fp32);

    return 0;
}

int ref_interp_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct interp_param* param)
{
    /* dequant */
    int input_total_size = input_tensor->elem_num;
    int output_total_size = output_tensor->elem_num;

    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* input_fp32 = (float*)sys_malloc(input_total_size * sizeof(float));
    float* output_fp32 = (float*)sys_malloc(output_total_size * sizeof(float));

    for (int i = 0; i < input_total_size; i++)
    {
        input_fp32[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    /* process */
    if (param->resize_type == 1)
    {
        int batch = output_tensor->dims[0];
        int channel = output_tensor->dims[1];
        int output_h = output_tensor->dims[2];
        int output_w = output_tensor->dims[3];
        int input_h = input_tensor->dims[2];
        int input_w = input_tensor->dims[3];

        for (int n = 0; n < batch; ++n)
        {
            for (int c = 0; c < channel; c++)
            {
                for (int h = 0; h < output_h; h++)
                {
                    for (int w = 0; w < output_w; w++)
                    {
                        int in_w = w / param->width_scale;
                        int in_h = h / param->height_scale;
                        int out_idx = n * channel * output_h * output_w + c * output_h * output_w + h * output_w + w;
                        int in_idx = n * channel * input_h * input_w + c * input_w * input_h + in_h * input_w + in_w;
                        output_fp32[out_idx] = input_fp32[in_idx];
                    }
                }
            }
        }
    }
    else if (param->resize_type == 2 || param->resize_type == 4)
    {
        int batch = input_tensor->dims[0];
        int channel = input_tensor->dims[1];
        int in_h = input_tensor->dims[2];
        int in_w = input_tensor->dims[3];
        int out_h = output_tensor->dims[2];
        int out_w = output_tensor->dims[3];

        int in_channel_size = in_h * in_w;
        int out_channel_size = out_h * out_w;

        int* buf = (int*)sys_malloc((param->output_width + param->output_height + param->output_width * 2 + param->output_height * 2) * sizeof(float));

        if (buf == NULL)
        {
            TLOG_ERR("interp malloc failed!\n");
            return -1;
        }

        int* xofs = buf;                       //new int[ow];
        int* yofs = buf + param->output_width; //new int[oh];

        float* alpha = (float*)(buf + param->output_width + param->output_height);                          //new float[ow * 2];
        float* beta = (float*)(buf + param->output_width + param->output_height + param->output_width * 2); //new float[oh * 2];

        int align_corner = param->resize_type == 2 ? 0 : 1;
        linear_coeffs(in_w, out_w, xofs, alpha, align_corner);
        linear_coeffs(in_h, out_h, yofs, beta, align_corner);

        for (int q = 0; q < channel; ++q)
        {
            resize_bilinear_image(input_fp32 + in_channel_size * q, output_fp32 + out_channel_size * q, alpha, xofs, beta, yofs, out_h, out_w, in_h, in_w);
        }

        sys_free(buf);
    }
    else
    {
        TLOG_ERR("interp resize type %d not support!\n", param->resize_type);
        return -1;
    }

    /* quant */
    for (int i = 0; i < output_total_size; i++)
    {
        int udata = round(output_fp32[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(input_fp32);
    sys_free(output_fp32);

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* graph = node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    struct interp_param* param = (struct interp_param*)node->op.param_mem;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_interp_fp32(input_tensor, output_tensor, param);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_interp_int8(input_tensor, output_tensor, param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_interp_uint8(input_tensor, output_tensor, param);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_interp_ref_op()
{
    return register_builtin_node_ops(OP_INTERP, &hcl_node_ops);
}

int unregister_interp_ref_op()
{
    return unregister_builtin_node_ops(OP_INTERP, &hcl_node_ops);
}
