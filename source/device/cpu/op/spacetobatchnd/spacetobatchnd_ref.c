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
 * Author: hhchen@openailab.com
 */

#include "spacetobatchnd_param.h"

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

static int ref_spacetobatchnd_fp32(struct tensor* input_tensor, struct tensor* output_tensor,
                                   struct spacetobatchnd_param* param, int num_thread)
{
    float* in_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    int out_dims[4];
    int in_dims[4];

    out_dims[0] = output_tensor->dims[0];
    out_dims[1] = output_tensor->dims[2];
    out_dims[2] = output_tensor->dims[3];
    out_dims[3] = output_tensor->dims[1];

    in_dims[0] = input_tensor->dims[0];
    in_dims[1] = input_tensor->dims[2];
    in_dims[2] = input_tensor->dims[3];
    in_dims[3] = input_tensor->dims[1];

    float* output_ptr = out_data;
    int output_batch_size = out_dims[0];
    int input_batch_size = in_dims[0];

    int block_shape_width = param->dilation_x;
    int block_shape_height = param->dilation_y;
    int output_height = out_dims[1];
    int output_width = out_dims[2];
    int padding_top = param->pad_top;

    int padding_left = param->pad_left;
    int input_height = in_dims[1];
    int input_width = in_dims[2];
    int depth = in_dims[3];

    const int out_stride_width = 1;
    const int out_stride_height = out_stride_width;
    const int out_stride_depth = out_stride_height * output_height;
    const int out_stride_batch = out_stride_depth * depth;

    const int in_stride_width = 1;
    const int in_stride_height = in_stride_width;
    const int in_stride_depth = in_stride_height * input_height;
    const int in_stride_batch = in_stride_depth * depth;

    for (int out_b = 0; out_b < output_batch_size; ++out_b)
    {
        int input_batch = out_b % input_batch_size;
        int shift_w = (out_b / input_batch_size) % block_shape_width;
        int shift_h = (out_b / input_batch_size) / block_shape_width;
        for (int c = 0; c < depth; ++c)
        {
            for (int out_h = 0; out_h < output_height; ++out_h)
            {
                for (int out_w = 0; out_w < output_width; ++out_w)
                {
                    float* out = out_data + out_b * out_stride_batch + c * out_stride_depth + out_h * out_stride_height + out_w;

                    if (out_h * block_shape_height + shift_h < padding_top || out_h * block_shape_height + shift_h >= padding_top + input_height || out_w * block_shape_width + shift_w < padding_left || out_w * block_shape_width + shift_w >= padding_left + input_width)
                    {
                        // This may not execute correctly when pad_value != 0 and T != uint8.
                        *out = 0;
                    }
                    else
                    {
                        const float* in = in_data + input_batch * in_stride_batch + c * in_stride_depth + ((out_h * block_shape_height + shift_h) - padding_top) * in_stride_height + ((out_w * block_shape_width + shift_w) - padding_left);
                        *out = *in;
                    }
                }
            }
        }
    }

    return 0;
}

static int ref_spacetobatchnd_uint8(struct tensor* input_tensor, struct tensor* output_tensor,
                                    struct spacetobatchnd_param* param, int num_thread)
{
    /* dequant */
    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* in_data = (float*)sys_malloc(input_size * sizeof(float));
    float* out_data = (float*)sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        in_data[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    int out_dims[4];
    int in_dims[4];

    out_dims[0] = output_tensor->dims[0];
    out_dims[1] = output_tensor->dims[2];
    out_dims[2] = output_tensor->dims[3];
    out_dims[3] = output_tensor->dims[1];

    in_dims[0] = input_tensor->dims[0];
    in_dims[1] = input_tensor->dims[2];
    in_dims[2] = input_tensor->dims[3];
    in_dims[3] = input_tensor->dims[1];

    float* output_ptr = out_data;
    int output_batch_size = out_dims[0];
    int input_batch_size = in_dims[0];

    int block_shape_width = param->dilation_x;
    int block_shape_height = param->dilation_y;
    int output_height = out_dims[1];
    int output_width = out_dims[2];
    int padding_top = param->pad_top;

    int padding_left = param->pad_left;
    int input_height = in_dims[1];
    int input_width = in_dims[2];
    int depth = in_dims[3];

    const int out_stride_width = 1;
    const int out_stride_height = out_stride_width;
    const int out_stride_depth = out_stride_height * output_height;
    const int out_stride_batch = out_stride_depth * depth;

    const int in_stride_width = 1;
    const int in_stride_height = in_stride_width;
    const int in_stride_depth = in_stride_height * input_height;
    const int in_stride_batch = in_stride_depth * depth;

    for (int out_b = 0; out_b < output_batch_size; ++out_b)
    {
        int input_batch = out_b % input_batch_size;
        int shift_w = (out_b / input_batch_size) % block_shape_width;
        int shift_h = (out_b / input_batch_size) / block_shape_width;
        for (int c = 0; c < depth; ++c)
        {
            for (int out_h = 0; out_h < output_height; ++out_h)
            {
                for (int out_w = 0; out_w < output_width; ++out_w)
                {
                    float* out = out_data + out_b * out_stride_batch + c * out_stride_depth + out_h * out_stride_height + out_w;

                    if (out_h * block_shape_height + shift_h < padding_top || out_h * block_shape_height + shift_h >= padding_top + input_height || out_w * block_shape_width + shift_w < padding_left || out_w * block_shape_width + shift_w >= padding_left + input_width)
                    {
                        // This may not execute correctly when pad_value != 0 and T != uint8.
                        *out = 0;
                    }
                    else
                    {
                        const float* in = in_data + input_batch * in_stride_batch + c * in_stride_depth + ((out_h * block_shape_height + shift_h) - padding_top) * in_stride_height + ((out_w * block_shape_width + shift_w) - padding_left);
                        *out = *in;
                    }
                }
            }
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(out_data[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(in_data);
    sys_free(out_data);

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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct spacetobatchnd_param* spacetobatchnd_param = (struct spacetobatchnd_param*)ir_node->op.param_mem;

    if (input_tensor->data_type == TENGINE_DT_FP32)
        ref_spacetobatchnd_fp32(input_tensor, output_tensor, spacetobatchnd_param, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ref_spacetobatchnd_uint8(input_tensor, output_tensor, spacetobatchnd_param, exec_graph->num_thread);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_spacetobatchnd_ref_op()
{
    return register_builtin_node_ops(OP_SPACETOBATCHND, &hcl_node_ops);
}

int unregister_spacetobatchnd_ref_op()
{
    return unregister_builtin_node_ops(OP_SPACETOBATCHND, &hcl_node_ops);
}
