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

#include "convolution_param.h"

#include "conv_dw_kernel_x86.h"

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

static void pad_int8(int8_t* input, int8_t* output, int in_h, int in_w, int out_h, int out_w, int top, int left, int8_t v)
{
    int8_t* ptr = input;
    int8_t* outptr = output;

    int y = 0;
    // fill top
    for (; y < top; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
    // fill center
    for (; y < (top + in_h); y++)
    {
        int x = 0;
        for (; x < left; x++)
        {
            outptr[x] = v;
        }
        if (in_w < 12)
        {
            for (; x < (left + in_w); x++)
            {
                outptr[x] = ptr[x - left];
            }
        }
        else
        {
            memcpy(outptr + left, ptr, in_w * sizeof(int8_t));
            x += in_w;
        }
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        ptr += in_w;
        outptr += out_w;
    }
    // fill bottom
    for (; y < out_h; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
}

static int convdw3x3s1_int8_sse(struct tensor* input_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor,
                                struct tensor* output_tensor, struct conv_param* param, int num_thread)
{
    int inch = input_tensor->dims[1];
    int inh = input_tensor->dims[2];
    int inw = input_tensor->dims[3];
    int in_hw = inh * inw;

    int outch = output_tensor->dims[1];
    int outh = output_tensor->dims[2];
    int outw = output_tensor->dims[3];
    int out_hw = outh * outw;
    int out_size = output_tensor->elem_num;

    int pad_w = param->pad_w0;
    int pad_h = param->pad_h0;

    int32_t* output_int32 = (int32_t*)sys_malloc(out_size * sizeof(int32_t));
    memset(output_int32, 0, out_size * sizeof(int32_t));
    float* output_fp32 = (float*)sys_malloc(out_size * sizeof(float));

    int8_t* output_int8 = (int8_t*)output_tensor->data;
    int8_t* input_int8 = (int8_t*)input_tensor->data;
    int32_t* bias_int32 = NULL;
    if (bias_tensor)
        bias_int32 = (int32_t*)bias_tensor->data;

    /* get scale value of quantizaiton */
    float input_scale = input_tensor->scale;
    float* kernel_scales = weight_tensor->scale_list;
    float output_scale = output_tensor->scale;

    const signed char* kernel = (const signed char*)weight_tensor->data;

    /* pading */
    int inh_tmp = inh + pad_h + pad_h;
    int inw_tmp = inw + pad_w + pad_w;
    int8_t* input_tmp = NULL;
    if (inh_tmp == inh && inw_tmp == inw)
        input_tmp = input_int8;
    else
    {
        input_tmp = (int8_t*)sys_malloc((size_t)inh_tmp * inw_tmp * inch * sizeof(int8_t));
#pragma omp parallel for num_threads(num_thread)
        for (int g = 0; g < inch; g++)
        {
            int8_t* pad_in = input_int8 + g * inh * inw;
            int8_t* pad_out = input_tmp + g * inh_tmp * inw_tmp;
            pad_int8(pad_in, pad_out, inh, inw, inh_tmp, inw_tmp, pad_h, pad_w, 0);
        }
    }

#pragma omp parallel for num_threads(num_thread)
    for (int p = 0; p < outch; p++)
    {
        int32_t* out0 = output_int32 + p * out_hw;
        int8_t* kernel0 = (int8_t*)kernel + p * 9;

        int* outptr0 = out0;

        int8_t* img0 = input_tmp + p * inw_tmp * inh_tmp;

        int8_t* r0 = img0;
        int8_t* r1 = img0 + inw_tmp;
        int8_t* r2 = img0 + inw_tmp * 2;

        for (int i = 0; i < outh; i++)
        {
            int remain = outw;

            for (; remain > 0; remain--)
            {
                int sum0 = 0;

                sum0 += (int)r0[0] * kernel0[0];
                sum0 += (int)r0[1] * kernel0[1];
                sum0 += (int)r0[2] * kernel0[2];
                sum0 += (int)r1[0] * kernel0[3];
                sum0 += (int)r1[1] * kernel0[4];
                sum0 += (int)r1[2] * kernel0[5];
                sum0 += (int)r2[0] * kernel0[6];
                sum0 += (int)r2[1] * kernel0[7];
                sum0 += (int)r2[2] * kernel0[8];

                *outptr0 += sum0;

                r0++;
                r1++;
                r2++;
                outptr0++;
            }

            r0 += 2;
            r1 += 2;
            r2 += 2;
        }

        kernel0 += 9;
    }

    /* process bias and dequant output from int32 to fp32 */
#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < outch; i++)
    {
        for (int j = 0; j < outh * outw; j++)
        {
            int output_off = i * (outh * outw) + j;
            if (bias_tensor)
                output_fp32[output_off] = (float)(output_int32[output_off] + bias_int32[i]) * input_scale * kernel_scales[i];
            else
                output_fp32[output_off] = (float)output_int32[output_off] * input_scale * kernel_scales[i];
        }
    }

    /* process activation relu */
    if (param->activation == 0)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < outch; i++)
        {
            for (int j = 0; j < outh * outw; j++)
            {
                int output_off = i * (outh * outw) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
            }
        }
    }

    /* process activation relu6 */
    if (param->activation > 0)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < outch; i++)
        {
            for (int j = 0; j < outh * outw; j++)
            {
                int output_off = i * (outh * outw) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
                if (output_fp32[output_off] > 6)
                    output_fp32[output_off] = 6;
            }
        }
    }

    /* quant from fp32 to int8 */
#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < outch; i++)
    {
        for (int j = 0; j < outh * outw; j++)
        {
            int output_off = i * (outh * outw) + j;

            int32_t data_i32 = (int32_t)(round(output_fp32[output_off] / output_scale));
            if (data_i32 > 127)
                data_i32 = 127;
            else if (data_i32 < -127)
                data_i32 = -127;
            output_int8[output_off] = (int8_t)data_i32;
        }
    }

    sys_free(output_int32);
    sys_free(output_fp32);

    if (!(inh_tmp == inh && inw_tmp == inw))
        sys_free(input_tmp);

    return 0;
}

static int convdw3x3s2_int8_sse(struct tensor* input_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor,
                                struct tensor* output_tensor, struct conv_param* param, int num_thread)
{
    int inch = input_tensor->dims[1];
    int inh = input_tensor->dims[2];
    int inw = input_tensor->dims[3];
    int in_hw = inh * inw;

    int outch = output_tensor->dims[1];
    int outh = output_tensor->dims[2];
    int outw = output_tensor->dims[3];
    int out_hw = outh * outw;
    int out_size = output_tensor->elem_num;

    int pad_w = param->pad_w0;
    int pad_h = param->pad_h0;

    int32_t* output_int32 = (int32_t*)sys_malloc(out_size * sizeof(int32_t));
    memset(output_int32, 0, out_size * sizeof(int32_t));
    float* output_fp32 = (float*)sys_malloc(out_size * sizeof(float));

    int8_t* output_int8 = (int8_t*)output_tensor->data;
    int8_t* input_int8 = (int8_t*)input_tensor->data;
    int32_t* bias_int32 = NULL;
    if (bias_tensor)
        bias_int32 = (int32_t*)bias_tensor->data;

    /* get scale value of quantizaiton */
    float input_scale = input_tensor->scale;
    float* kernel_scales = weight_tensor->scale_list;
    float output_scale = output_tensor->scale;

    const signed char* kernel = (const signed char*)weight_tensor->data;

    /* pading */
    int inh_tmp = inh + pad_h + pad_h;
    int inw_tmp = inw + pad_w + pad_w;
    int8_t* input_tmp = NULL;
    if (inh_tmp == inh && inw_tmp == inw)
        input_tmp = input_int8;
    else
    {
        input_tmp = (int8_t*)sys_malloc((size_t)inh_tmp * inw_tmp * inch * sizeof(int8_t));
#pragma omp parallel for num_threads(num_thread)
        for (int g = 0; g < inch; g++)
        {
            int8_t* pad_in = input_int8 + g * inh * inw;
            int8_t* pad_out = input_tmp + g * inh_tmp * inw_tmp;
            pad_int8(pad_in, pad_out, inh, inw, inh_tmp, inw_tmp, pad_h, pad_w, 0);
        }
    }

    int tailstep = inw_tmp - 2 * outw + inw_tmp;

#pragma omp parallel for num_threads(num_thread)
    for (int p = 0; p < outch; p++)
    {
        int32_t* out0 = output_int32 + p * out_hw;
        int8_t* kernel0 = (int8_t*)kernel + p * 9;

        int* outptr0 = out0;

        int8_t* img0 = input_tmp + p * inw_tmp * inh_tmp;

        int8_t* r0 = img0;
        int8_t* r1 = img0 + inw_tmp;
        int8_t* r2 = img0 + inw_tmp * 2;

        for (int i = 0; i < outh; i++)
        {
            int remain = outw;

            for (; remain > 0; remain--)
            {
                int sum0 = 0;

                sum0 += (int)r0[0] * kernel0[0];
                sum0 += (int)r0[1] * kernel0[1];
                sum0 += (int)r0[2] * kernel0[2];
                sum0 += (int)r1[0] * kernel0[3];
                sum0 += (int)r1[1] * kernel0[4];
                sum0 += (int)r1[2] * kernel0[5];
                sum0 += (int)r2[0] * kernel0[6];
                sum0 += (int)r2[1] * kernel0[7];
                sum0 += (int)r2[2] * kernel0[8];

                *outptr0 += sum0;

                r0 += 2;
                r1 += 2;
                r2 += 2;
                outptr0++;
            }

            r0 += tailstep;
            r1 += tailstep;
            r2 += tailstep;
        }

        kernel0 += 9;
    }

    /* process bias and dequant output from int32 to fp32 */
#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < outch; i++)
    {
        for (int j = 0; j < outh * outw; j++)
        {
            int output_off = i * (outh * outw) + j;
            if (bias_tensor)
                output_fp32[output_off] = (float)(output_int32[output_off] + bias_int32[i]) * input_scale * kernel_scales[i];
            else
                output_fp32[output_off] = (float)output_int32[output_off] * input_scale * kernel_scales[i];
        }
    }

    /* process activation relu */
    if (param->activation == 0)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < outch; i++)
        {
            for (int j = 0; j < outh * outw; j++)
            {
                int output_off = i * (outh * outw) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
            }
        }
    }

    /* process activation relu6 */
    if (param->activation > 0)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < outch; i++)
        {
            for (int j = 0; j < outh * outw; j++)
            {
                int output_off = i * (outh * outw) + j;

                if (output_fp32[output_off] < 0)
                    output_fp32[output_off] = 0;
                if (output_fp32[output_off] > 6)
                    output_fp32[output_off] = 6;
            }
        }
    }

    /* quant from fp32 to int8 */
#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < outch; i++)
    {
        for (int j = 0; j < outh * outw; j++)
        {
            int output_off = i * (outh * outw) + j;

            int32_t data_i32 = (int32_t)(round(output_fp32[output_off] / output_scale));
            if (data_i32 > 127)
                data_i32 = 127;
            else if (data_i32 < -127)
                data_i32 = -127;
            output_int8[output_off] = (int8_t)data_i32;
        }
    }

    sys_free(output_int32);
    sys_free(output_fp32);

    if (!(inh_tmp == inh && inw_tmp == inw))
        sys_free(input_tmp);

    return 0;
}

static int conv_dw_run_int8(struct tensor* input_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor,
                            struct tensor* output_tensor, struct conv_param* param, int num_thread)
{
    int ret = -1;
    switch (param->stride_h)
    {
    case 1:
        ret = convdw3x3s1_int8_sse(input_tensor, weight_tensor, bias_tensor, output_tensor, param, num_thread);
        break;
    case 2:
        ret = convdw3x3s2_int8_sse(input_tensor, weight_tensor, bias_tensor, output_tensor, param, num_thread);
        break;
    default:
        TLOG_ERR("Direct Convolution Int8 not support the stride %d\n", param->stride_h);
    }

    return ret;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* bias_tensor = NULL;
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);

    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = (struct conv_priv_info*)exec_node->ops_priv;

    int ret = -1;
    if (exec_graph->mode == TENGINE_MODE_FP32)
        ret = conv_dw_run(input_tensor, weight_tensor, bias_tensor, output_tensor, conv_priv_info, conv_param, num_thread, cpu_affinity);
    else if (exec_graph->mode == TENGINE_MODE_INT8)
        ret = conv_dw_run_int8(input_tensor, weight_tensor, bias_tensor, output_tensor, conv_param, num_thread);
    else
    {
        TLOG_ERR("hcl conv run failed\n");
        return -1;
    }

    return ret;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct conv_param* param = (struct conv_param*)exec_node->op.param_mem;
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    int pad_h0 = param->pad_h0;
    int pad_w0 = param->pad_w0;
    int pad_h1 = param->pad_h1;
    int pad_w1 = param->pad_w1;

    int in_c = input_tensor->dims[1] / group;
    int out_c = output_tensor->dims[1] / group;

    /* todo support uint8 */
    if (!(input_tensor->data_type == TENGINE_DT_FP32 || input_tensor->data_type == TENGINE_DT_INT8))
        return 0;

    if (kernel_h != kernel_w || input_tensor->dims[0] > 1)
        return 0;

    if (param->group > 1 && in_c == 1 && out_c == 1 && pad_h0 == pad_h1 && pad_w0 == pad_w1 && dilation_h == 1 && dilation_w == 1 && kernel_h == 3 && kernel_w == 3 && ((stride_h == 1 && stride_w == 1) || (stride_h == 2 && stride_w == 2)))
        return OPS_SCORE_BEST;
    else
        return 0;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = false};

int register_conv_dw_hcl_x86_op()
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

int unregister_conv_dw_hcl_x86_op()
{
    unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
    return 0;
}
