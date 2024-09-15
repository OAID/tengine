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

#include "upsample_param.h"

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

static int ref_upsample_fp32(struct tensor* input_tensor, struct tensor* output_tensor,
                             struct upsample_param* param, int num_thread)
{
    float* input = (float*)input_tensor->data;
    float* output = (float*)output_tensor->data;

    float scale = param->scale;
    int batch = output_tensor->dims[0];
    int channel = output_tensor->dims[1];
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];

    for (int n = 0; n < batch; ++n)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int h = 0; h < out_h; h++)
            {
                for (int w = 0; w < out_w; w++)
                {
                    int in_w = w / scale;
                    int in_h = h / scale;
                    int out_idx = n * channel * out_h * out_w + c * out_h * out_w + h * out_w + w;
                    int in_idx = n * channel * input_h * input_w + c * input_w * input_h + in_h * input_w + in_w;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }

    return 0;
}

static int ref_upsample_uint8(struct tensor* input_tensor, struct tensor* output_tensor,
                              struct upsample_param* param, int num_thread)
{
    float scale = param->scale;
    int batch = output_tensor->dims[0];
    int channel = output_tensor->dims[1];
    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];

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

    /* fp32 inference */
    for (int n = 0; n < batch; ++n)
    {
        for (int c = 0; c < channel; c++)
        {
            for (int h = 0; h < out_h; h++)
            {
                for (int w = 0; w < out_w; w++)
                {
                    int in_w = w / scale;
                    int in_h = h / scale;
                    int out_idx = n * channel * out_h * out_w + c * out_h * out_w + h * out_w + w;
                    int in_idx = n * channel * input_h * input_w + c * input_w * input_h + in_h * input_w + in_w;
                    output_fp32[out_idx] = input_fp32[in_idx];
                }
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
    struct tensor* roi_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct upsample_param* upsample_param = (struct upsample_param*)ir_node->op.param_mem;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_upsample_fp32(input_tensor, output_tensor, upsample_param, exec_graph->num_thread);
    else
        ret = ref_upsample_uint8(input_tensor, output_tensor, upsample_param, exec_graph->num_thread);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
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

int register_upsample_ref_op()
{
    return register_builtin_node_ops(OP_UPSAMPLE, &hcl_node_ops);
}

int unregister_upsample_ref_op()
{
    return unregister_builtin_node_ops(OP_UPSAMPLE, &hcl_node_ops);
}
