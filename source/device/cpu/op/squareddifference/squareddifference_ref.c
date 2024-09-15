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

int ref_squareddifference_fp32(struct tensor* input_tensor_0, struct tensor* input_tensor_1,
                               struct tensor* output_tensor, int num_thread)
{
    // dims size = 2 or 3
    if (input_tensor_0->dim_num < 4)
    {
        float* input0 = (float*)input_tensor_0->data;
        float* input1 = (float*)input_tensor_1->data;
        float* output = (float*)output_tensor->data;
        int total_size = output_tensor->elem_num;

        for (int i = 0; i < total_size; i++)
        {
            output[i] = powf((input0[i] - input1[i]), 2);
        }

        return 0;
    }
    // dims size 3
    else if (output_tensor->dim_num == 4)
    {
        int w = output_tensor->dims[3];
        int h = output_tensor->dims[2];
        int channels = output_tensor->dims[1];
        int size = h * w;
        int c_step = h * w;

        float* input0 = (float*)input_tensor_0->data;
        float* input1 = (float*)input_tensor_1->data;
        float* output = (float*)output_tensor->data;

#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < channels; q++)
        {
            float* src0 = input0 + c_step * q;
            float* src1 = input1 + c_step * q;
            float* dst = output + c_step * q;

            for (int i = 0; i < size; i++)
            {
                dst[i] = powf((src0[i] - src1[i]), 2);
            }
        }

        return 0;
    }

    return -1;
}

int ref_squareddifference_uint8(struct tensor* input_tensor_0, struct tensor* input_tensor_1,
                                struct tensor* output_tensor, int num_thread)
{
    /* dequant */
    uint8_t* input0_uint8 = (uint8_t*)input_tensor_0->data;
    uint8_t* input1_uint8 = (uint8_t*)input_tensor_1->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input0_scale = input_tensor_0->scale;
    float input1_scale = input_tensor_1->scale;
    float output_scale = output_tensor->scale;
    int32_t input0_zero = input_tensor_0->zero_point;
    int32_t input1_zero = input_tensor_1->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input0_size = input_tensor_0->elem_num;
    int input1_size = input_tensor_1->elem_num;
    int output_size = output_tensor->elem_num;

    float* input0 = (float*)sys_malloc(input0_size * sizeof(float));
    float* input1 = (float*)sys_malloc(input1_size * sizeof(float));
    float* output = (float*)sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input0_size; i++)
    {
        input0[i] = ((float)input0_uint8[i] - (float)input0_zero) * input0_scale;
    }
    for (int i = 0; i < input1_size; i++)
    {
        input1[i] = ((float)input1_uint8[i] - (float)input1_zero) * input1_scale;
    }

    // dims size = 2 or 3
    if (input_tensor_0->dim_num < 4)
    {
        int total_size = output_tensor->elem_num;

        for (int i = 0; i < total_size; i++)
        {
            output[i] = powf((input0[i] - input1[i]), 2);
        }

        return 0;
    }
    // dims size 3
    else if (output_tensor->dim_num == 4)
    {
        int w = output_tensor->dims[3];
        int h = output_tensor->dims[2];
        int channels = output_tensor->dims[1];
        int size = h * w;
        int c_step = h * w;

#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < channels; q++)
        {
            float* src0 = input0 + c_step * q;
            float* src1 = input1 + c_step * q;
            float* dst = output + c_step * q;

            for (int i = 0; i < size; i++)
            {
                dst[i] = powf((src0[i] - src1[i]), 2);
            }
        }

        return 0;
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(output[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(input0);
    sys_free(input1);
    sys_free(output);

    return -1;
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
    struct tensor* input_tensor_0;
    struct tensor* input_tensor_1;
    struct tensor* output_tensor;
    int layout = ir_graph->graph_layout;

    input_tensor_0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    input_tensor_1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int ret = -1;
    if (input_tensor_0->data_type == TENGINE_DT_FP32)
        ret = ref_squareddifference_fp32(input_tensor_0, input_tensor_1, output_tensor, exec_graph->num_thread);
    else if (input_tensor_0->data_type == TENGINE_DT_UINT8)
        ret = ref_squareddifference_uint8(input_tensor_0, input_tensor_1, output_tensor, exec_graph->num_thread);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
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

int register_squareddifference_ref_op()
{
    return register_builtin_node_ops(OP_SQUAREDDIFFERENCE, &hcl_node_ops);
}

int unregister_squareddifference_ref_op()
{
    return unregister_builtin_node_ops(OP_SQUAREDDIFFERENCE, &hcl_node_ops);
}
