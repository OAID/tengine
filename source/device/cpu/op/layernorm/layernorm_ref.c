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
 * Author: Shijie Chen
 */

#include "layernorm_param.h"

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

static int ref_layernorm_fp32(struct tensor* input_tensor, struct tensor* output_tensor,
                              struct tensor* gamma_tensor, struct tensor* beta_tensor, float eps)
{
#if 1
    // TIM-VX
    int norm_size = input_tensor->dims[input_tensor->dim_num - 1];
    int count = 1;
    for (int i = 0; i < input_tensor->dim_num - 1; i++)
    {
        count *= input_tensor->dims[i];
    }
#else
    // PyTorch
    int norm_size = gamma_tensor->elem_num;
    int count = input_tensor->elem_num / gamma_tensor->elem_num;
#endif

    const float* input_data = (const float*)input_tensor->data;
    float* output_data = (float*)output_tensor->data;

    const float* gamma_data = (const float*)gamma_tensor->data;
    const float* beta_data = (const float*)beta_tensor->data;

    for (int i = 0; i < count; i++)
    {
        float sum = 0.f;
        float sqsum = 0.f;
        for (int j = 0; j < norm_size; j++)
        {
            float x = input_data[i * norm_size + j];
            sum += x;
            sqsum += x * x;
        }
        float mean = sum / norm_size;
        float var = sqsum / norm_size - mean * mean;
        float a = 1.0f / sqrtf(var + eps);
        float b = -mean * a;
        for (int j = 0; j < norm_size; j++)
        {
            int offset = i * norm_size + j;
            output_data[offset] = (input_data[offset] * a + b) * gamma_data[j] + beta_data[j];
        }
    }

    return 0;
}

static int ref_layernorm_uint8(struct tensor* input_tensor, struct tensor* output_tensor,
                               struct tensor* gamma_tensor, struct tensor* beta_tensor, float eps)
{
#if 1
    // TIM-VX
    int norm_size = input_tensor->dims[input_tensor->dim_num - 1];
    int count = 1;
    for (int i = 0; i < input_tensor->dim_num - 1; i++)
    {
        count *= input_tensor->dims[i];
    }
#else
    // PyTorch
    int norm_size = gamma_tensor->elem_num;
    int count = input_tensor->elem_num / gamma_tensor->elem_num;
#endif

    int total_size = input_tensor->elem_num;
    float* input_data = (float*)sys_malloc(total_size * sizeof(float));
    float* output_data = (float*)sys_malloc(total_size * sizeof(float));

    // dequant
    {
        const uint8_t* input_uint8 = (const uint8_t*)input_tensor->data;
        float input_scale = input_tensor->scale;
        int input_zero = input_tensor->zero_point;

        for (int i = 0; i < total_size; i++)
            input_data[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    const float* gamma_data = (const float*)gamma_tensor->data;
    const float* beta_data = (const float*)beta_tensor->data;

    for (int i = 0; i < count; i++)
    {
        float sum = 0.f;
        float sqsum = 0.f;
        for (int j = 0; j < norm_size; j++)
        {
            float x = input_data[i * norm_size + j];
            sum += x;
            sqsum += x * x;
        }
        float mean = sum / norm_size;
        float var = sqsum / norm_size - mean * mean;
        float a = 1.0f / sqrtf(var + eps);
        float b = -mean * a;
        for (int j = 0; j < norm_size; j++)
        {
            int offset = i * norm_size + j;
            output_data[offset] = (input_data[offset] * a + b) * gamma_data[j] + beta_data[j];
        }
    }

    // quant
    {
        uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
        float output_scale = output_tensor->scale;
        int output_zero = output_tensor->zero_point;
        for (int i = 0; i < total_size; i++)
        {
            int udata = (int)roundf(output_data[i] / output_scale + output_zero);
            if (udata > 255)
                udata = 255;
            else if (udata < 0)
                udata = 0;
            output_uint8[i] = udata;
        }
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

    struct layernorm_Param* param = (struct layernorm_Param*)node->op.param_mem;
    float eps = param->eps;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_layernorm_fp32(input_tensor, output_tensor, gamma_tensor, beta_tensor, eps);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_layernorm_uint8(input_tensor, output_tensor, gamma_tensor, beta_tensor, eps);

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

int register_layernorm_ref_op()
{
    return register_builtin_node_ops(OP_LAYERNORM, &hcl_node_ops);
}

int unregister_layernorm_ref_op()
{
    return unregister_builtin_node_ops(OP_LAYERNORM, &hcl_node_ops);
}
