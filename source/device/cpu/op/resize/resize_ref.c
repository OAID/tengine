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

#include "resize_param.h"

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

#define T_MAX(a, b) ((a) > (b) ? (a) : (b))
#define T_MIN(a, b) ((a) < (b) ? (a) : (b))

static void bilinear_resize(float* inp, float* output, int h, int w, int c, float scale_x, float scale_y, int oh,
                            int ow)
{
    int out_hw = oh * ow;
    int in_hw = h * w;
    for (int j = 0; j < oh; j++)
    {
        float fy = (j + 0.5) * scale_y - 0.5;
        int sy = floor(fy);
        fy -= sy;
        sy = T_MIN(sy, h - 2);
        sy = T_MAX(0, sy);
        float fy_0 = 1.f - fy;

        for (int i = 0; i < ow; i++)
        {
            float fx = (i + 0.5) * scale_x - 0.5;
            int sx = floor(fx);
            fx -= sx;
            if (sx < 0)
            {
                sx = 0;
                fx = 0;
            }
            if (sx >= w - 1)
            {
                fx = 0;
                sx = w - 2;
            }
            float fx_0 = 1.f - fx;
            int out_idx = j * ow + i;
            int in_idx = sy * w + sx;
            for (int k = 0; k < c; k++)
            {
                int in_index = in_idx + k * in_hw;
                output[k * out_hw + out_idx] = inp[in_index] * fx_0 * fy_0 + inp[in_index + w] * fx_0 * fy + inp[in_index + 1] * fx * fy_0 + inp[in_index + w + 1] * fx * fy;
            }
        }
    }
}

static void bilinear_resize_int8(struct tensor* input_tensor, struct tensor* output_tensor, float scale_x, float scale_y)
{
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int c = input_tensor->dims[1];
    int oh = output_tensor->dims[2];
    int ow = output_tensor->dims[3];
    int out_hw = oh * ow;
    int in_hw = h * w;

    /* dequant */
    int8_t* input_int8 = (int8_t*)input_tensor->data;
    int8_t* output_int8 = (int8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* input_fp32 = (float*)sys_malloc(input_size * sizeof(float));
    float* output_fp32 = (float*)sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input_fp32[i] = ((float)input_int8[i] - (float)input_zero) * input_scale;
    }

    for (int j = 0; j < oh; j++)
    {
        float fy = (j + 0.5) * scale_y - 0.5;
        int sy = floor(fy);
        fy -= sy;
        sy = T_MIN(sy, h - 2);
        sy = T_MAX(0, sy);
        float fy_0 = 1.f - fy;

        for (int i = 0; i < ow; i++)
        {
            float fx = (i + 0.5) * scale_x - 0.5;
            int sx = floor(fx);
            fx -= sx;
            if (sx < 0)
            {
                sx = 0;
                fx = 0;
            }
            if (sx >= w - 1)
            {
                fx = 0;
                sx = w - 2;
            }
            float fx_0 = 1.f - fx;
            int out_idx = j * ow + i;
            int in_idx = sy * w + sx;
            for (int k = 0; k < c; k++)
            {
                int in_index = in_idx + k * in_hw;
                output_fp32[k * out_hw + out_idx] = input_fp32[in_index] * fx_0 * fy_0 + input_fp32[in_index + w] * fx_0 * fy + input_fp32[in_index + 1] * fx * fy_0 + input_fp32[in_index + w + 1] * fx * fy;
            }
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
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
}

static void bilinear_resize_uint8(struct tensor* input_tensor, struct tensor* output_tensor, float scale_x, float scale_y)
{
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int c = input_tensor->dims[1];
    int oh = output_tensor->dims[2];
    int ow = output_tensor->dims[3];
    int out_hw = oh * ow;
    int in_hw = h * w;

    /* dequant */
    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* input_fp32 = (float*)sys_malloc(input_size * sizeof(float));
    float* output_fp32 = (float*)sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input_fp32[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    for (int j = 0; j < oh; j++)
    {
        float fy = (j + 0.5) * scale_y - 0.5;
        int sy = floor(fy);
        fy -= sy;
        sy = T_MIN(sy, h - 2);
        sy = T_MAX(0, sy);
        float fy_0 = 1.f - fy;

        for (int i = 0; i < ow; i++)
        {
            float fx = (i + 0.5) * scale_x - 0.5;
            int sx = floor(fx);
            fx -= sx;
            if (sx < 0)
            {
                sx = 0;
                fx = 0;
            }
            if (sx >= w - 1)
            {
                fx = 0;
                sx = w - 2;
            }
            float fx_0 = 1.f - fx;
            int out_idx = j * ow + i;
            int in_idx = sy * w + sx;
            for (int k = 0; k < c; k++)
            {
                int in_index = in_idx + k * in_hw;
                output_fp32[k * out_hw + out_idx] = input_fp32[in_index] * fx_0 * fy_0 + input_fp32[in_index + w] * fx_0 * fy + input_fp32[in_index + 1] * fx * fy_0 + input_fp32[in_index + w + 1] * fx * fy;
            }
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
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
}

static void nearest_neighbor_resize(float* inp, float* out, int h, int w, int c_start, int c_end, float scale_x,
                                    float scale_y, int oh, int ow)
{
    float* output;
    float* input;
    int si, sj;
    for (int k = c_start; k < c_end; k++)
    {
        input = inp + k * h * w;
        output = out + k * oh * ow;
        for (int i = 0; i < oh; i++)
        {
            si = T_MIN((int)(i * scale_y), h - 1);
            for (int j = 0; j < ow; j++)
            {
                sj = T_MIN((int)(j * scale_x), w - 1);
                output[i * ow + j] = input[si * w + sj];
            }
        }
    }
}

static void nearest_neighbor_resize_int8(struct tensor* input_tensor, struct tensor* output_tensor, float scale_x, float scale_y)
{
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int c_start = 0;
    int c_end = input_tensor->dims[1];
    int oh = output_tensor->dims[2];
    int ow = output_tensor->dims[3];

    /* dequant */
    int8_t* input_int8 = (int8_t*)input_tensor->data;
    int8_t* output_int8 = (int8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* input_fp32 = (float*)sys_malloc(input_size * sizeof(float));
    float* output_fp32 = (float*)sys_malloc(output_size * sizeof(float));

    float* input_temp = NULL;
    float* output_temp = NULL;

    for (int i = 0; i < input_size; i++)
    {
        input_fp32[i] = ((float)input_int8[i] - (float)input_zero) * input_scale;
    }

    int si, sj;
    for (int k = c_start; k < c_end; k++)
    {
        input_temp = input_fp32 + k * h * w;
        output_temp = output_fp32 + k * oh * ow;
        for (int i = 0; i < oh; i++)
        {
            si = T_MIN((int)(i * scale_y), h - 1);
            for (int j = 0; j < ow; j++)
            {
                sj = T_MIN((int)(j * scale_x), w - 1);
                output_temp[i * ow + j] = input_temp[si * w + sj];
            }
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
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
}

static void nearest_neighbor_resize_uint8(struct tensor* input_tensor, struct tensor* output_tensor, float scale_x, float scale_y)
{
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int c_start = 0;
    int c_end = input_tensor->dims[1];
    int oh = output_tensor->dims[2];
    int ow = output_tensor->dims[3];

    /* dequant */
    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* input_fp32 = (float*)sys_malloc(input_size * sizeof(float));
    float* output_fp32 = (float*)sys_malloc(output_size * sizeof(float));

    float* input_temp = NULL;
    float* output_temp = NULL;

    for (int i = 0; i < input_size; i++)
    {
        input_fp32[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    int si, sj;
    for (int k = c_start; k < c_end; k++)
    {
        input_temp = input_fp32 + k * h * w;
        output_temp = output_fp32 + k * oh * ow;
        for (int i = 0; i < oh; i++)
        {
            si = T_MIN((int)(i * scale_y), h - 1);
            for (int j = 0; j < ow; j++)
            {
                sj = T_MIN((int)(j * scale_x), w - 1);
                output_temp[i * ow + j] = input_temp[si * w + sj];
            }
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
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
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct resize_param* resize_param = (struct resize_param*)ir_node->op.param_mem;

    float scale_x = 1.f / resize_param->scale_w;
    float scale_y = 1.f / resize_param->scale_h;
    int in_chw = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];
    int out_chw = output_tensor->dims[1] * output_tensor->dims[2] * output_tensor->dims[3];
    float* input = (float*)input_tensor->data;
    float* output = (float*)output_tensor->data;

    if (input_tensor->data_type == TENGINE_DT_FP32)
    {
        if (resize_param->type == 0)
        {
            for (int i = 0; i < input_tensor->dims[0]; i++)
            {
                nearest_neighbor_resize(input, output, input_tensor->dims[2], input_tensor->dims[3], 0,
                                        input_tensor->dims[1], scale_x, scale_y, output_tensor->dims[2],
                                        output_tensor->dims[3]);
                input += in_chw;
                output += out_chw;
            }
        }
        else
        {
            for (int i = 0; i < input_tensor->dims[0]; i++)
            {
                bilinear_resize(input, output, input_tensor->dims[2], input_tensor->dims[3], input_tensor->dims[1], scale_x,
                                scale_y, output_tensor->dims[2], output_tensor->dims[3]);
                input_tensor += in_chw;
                output_tensor += out_chw;
            }
        }
    }
    else if (input_tensor->data_type == TENGINE_DT_INT8)
    {
        if (resize_param->type == 0)
        {
            for (int i = 0; i < input_tensor->dims[0]; i++)
            {
                nearest_neighbor_resize_int8(input_tensor, output_tensor, scale_x, scale_y);
                input += in_chw;
                output += out_chw;
            }
        }
        else
        {
            for (int i = 0; i < input_tensor->dims[0]; i++)
            {
                bilinear_resize_int8(input_tensor, output_tensor, scale_x, scale_y);
                input_tensor += in_chw;
                output_tensor += out_chw;
            }
        }
    }
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
    {
        if (resize_param->type == 0)
        {
            for (int i = 0; i < input_tensor->dims[0]; i++)
            {
                nearest_neighbor_resize_uint8(input_tensor, output_tensor, scale_x, scale_y);
                input += in_chw;
                output += out_chw;
            }
        }
        else
        {
            for (int i = 0; i < input_tensor->dims[0]; i++)
            {
                bilinear_resize_uint8(input_tensor, output_tensor, scale_x, scale_y);
                input_tensor += in_chw;
                output_tensor += out_chw;
            }
        }
    }
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_resize_ref_op()
{
    return register_builtin_node_ops(OP_RESIZE, &hcl_node_ops);
}

int unregister_resize_ref_op()
{
    return unregister_builtin_node_ops(OP_RESIZE, &hcl_node_ops);
}
