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

#define SIGMOID_MAX(a, b) ((a) > (b) ? (a) : (b))
#define SIGMOID_MIN(a, b) ((a) < (b) ? (a) : (b))

int ref_sigmoid_fp32(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread)
{
    int dim_num = input_tensor->dim_num;

    if (dim_num == 4)
    {
        int batch = input_tensor->dims[0];
        int channel = input_tensor->dims[1];
        int cstep = input_tensor->dims[2] * input_tensor->dims[3];
        int bstep = channel * cstep;

        for (int n = 0; n < batch; n++)
        {
#pragma omp parallel for num_threads(num_thread)
            for (int c = 0; c < channel; c++)
            {
                float* input_data = (float*)input_tensor->data + n * bstep + c * cstep;
                float* output_data = (float*)output_tensor->data + n * bstep + c * cstep;
                for (int i = 0; i < cstep; i++)
                {
                    output_data[i] = SIGMOID_MIN(input_data[i], 30.0f);
                    output_data[i] = SIGMOID_MAX(input_data[i], -30.0f);
                    output_data[i] = 1.f / (1 + expf(-output_data[i]));
                }
            }
        }
    }
    else
    {
        uint32_t elem_num = input_tensor->elem_num;
        float* input_data = (float*)input_tensor->data;
        float* output_data = (float*)output_tensor->data;

        for (int i = 0; i < elem_num; i++)
        {
            output_data[i] = SIGMOID_MIN(input_data[i], 30.0f);
            output_data[i] = SIGMOID_MAX(input_data[i], -30.0f);
            output_data[i] = 1.f / (1 + expf(-output_data[i]));
        }
    }

    return 0;
}

int ref_sigmoid_int8(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread)
{
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

    for (int i = 0; i < input_size; i++)
    {
        output_fp32[i] = SIGMOID_MIN(input_fp32[i], 30.0f);
        output_fp32[i] = SIGMOID_MAX(input_fp32[i], -30.0f);

        output_fp32[i] = 1 / (1 + exp(-output_fp32[i]));
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

    return 0;
}

int ref_sigmoid_uint8(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread)
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

    float* input_fp32 = (float*)sys_malloc(input_size * sizeof(float));
    float* output_fp32 = (float*)sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input_fp32[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    for (int i = 0; i < input_size; i++)
    {
        output_fp32[i] = SIGMOID_MIN(input_fp32[i], 30.0f);
        output_fp32[i] = SIGMOID_MAX(input_fp32[i], -30.0f);

        output_fp32[i] = 1 / (1 + exp(-output_fp32[i]));
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

static int reshape_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    int ret = 0;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (input_tensor->dims[1] != output_tensor->dims[1] || input_tensor->dims[2] != output_tensor->dims[2] || input_tensor->dims[3] != output_tensor->dims[3])
        ret = set_ir_tensor_shape(output_tensor, input_tensor->dims, input_tensor->dim_num);

    return ret;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_sigmoid_fp32(input_tensor, output_tensor, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_sigmoid_int8(input_tensor, output_tensor, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_sigmoid_uint8(input_tensor, output_tensor, exec_graph->num_thread);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops sigmoid_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = reshape_node,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_sigmoid_ref_op()
{
    return register_builtin_node_ops(OP_SIGMOID, &sigmoid_node_ops);
}

int unregister_sigmoid_ref_op()
{
    return unregister_builtin_node_ops(OP_SIGMOID, &sigmoid_node_ops);
}
