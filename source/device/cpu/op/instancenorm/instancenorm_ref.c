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
 * Author: zpluo@openailab.com
 */

#include "instancenorm_param.h"

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

int ref_instancenorm_fp32(float* input_data, float* output_data, float* gamma_data, float* beta_data, int size,
                          int channels, int n, float eps, float scale, float zero_point, int layout)
{
    int image_size = channels * size;
    for (int s = 0; s < n; s++)
    {
        for (int i = 0; i < channels; i++)
        {
            float sum = 0.f;
            float sqsum = 0.f;
            int offset = 0;
            for (int j = 0; j < size; j++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                sum += input_data[offset];
            }
            float mean = sum / size;
            float tmp = 0.f;
            for (int j = 0; j < size; j++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                tmp = input_data[offset] - mean;
                sqsum += tmp * tmp;
            }
            float var = sqsum / size;

            float a = gamma_data[i] / (sqrt(var + eps));
            float b = -mean * a + beta_data[i];
            for (int j = 0; j < size; j++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                output_data[offset] = input_data[offset] * a + b;
            }
        }
    }
    return 0;
}

int ref_instancenorm_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* gamma_tensor, struct tensor* beta_tensor,
                           float eps, float scale, float zero_point, int layout)
{
    int n = input_tensor->dims[0];
    int channels = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int size = w * h;
    int image_size = channels * size;
    int total_size = image_size * n;

    float* beta_data = (float*)beta_tensor->data;
    float* gamma_data = (float*)gamma_tensor->data;

    // dequant
    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* input_data = (float*)sys_malloc(total_size * sizeof(float));
    float* output_data = (float*)sys_malloc(total_size * sizeof(float));
    for (int i = 0; i < total_size; i++)
        input_data[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;

    for (int s = 0; s < n; s++)
    {
        for (int i = 0; i < channels; i++)
        {
            float sum = 0.f;
            float sqsum = 0.f;
            int offset = 0;
            for (int j = 0; j < size; j++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                sum += input_data[offset];
            }
            float mean = sum / size;
            float tmp = 0.f;
            for (int j = 0; j < size; j++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                tmp = input_data[offset] - mean;
                sqsum += tmp * tmp;
            }
            float var = sqsum / size;

            float a = gamma_data[i] / (sqrt(var + eps));
            float b = -mean * a + beta_data[i];
            for (int j = 0; j < size; j++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = s * image_size + i * size + j;
                else
                    offset = s * image_size + j * channels + i;
                output_data[offset] = input_data[offset] * a + b;
            }
        }
    }

    // quant
    for (int i = 0; i < total_size; i++)
    {
        int udata = (int)roundf(output_data[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(input_data);
    sys_free(output_data);
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* graph = node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* gamma_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* beta_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);

    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int n = input_tensor->dims[0];
    int c = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];

    if (w == 0)
    {
        w = 1;
    }

    int size = w * h;

    void* in_data = input_tensor->data;
    void* out_data = output_tensor->data;
    void* beta_data = beta_tensor->data;
    void* gamma_data = gamma_tensor->data;

    struct instancenorm_Param* param = (struct instancenorm_Param*)node->op.param_mem;
    float eps = param->eps;
    float scale = 1.f;
    int zero_point = 0;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_instancenorm_fp32((float*)in_data, (float*)out_data, (float*)gamma_data, (float*)beta_data, size, c, n, eps, scale, zero_point, 0);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_instancenorm_uint8(input_tensor, output_tensor, gamma_tensor, beta_tensor, eps, scale, zero_point, 0);

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

int register_instancenorm_ref_op()
{
    return register_builtin_node_ops(OP_INSTANCENORM, &hcl_node_ops);
}

int unregister_instancenorm_ref_op()
{
    return unregister_builtin_node_ops(OP_INSTANCENORM, &hcl_node_ops);
}
