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
 * Update: hhchen@openailab.com
 */

#include "elu_param.h"

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

typedef struct __elu_param
{
    float scale;
    int zero_point;
    float alpha;
} _elu_param, *p_elu_param;

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

int ref_elu_fp32(float* data, float* out_data, int size, p_elu_param param)
{
    for (int i = 0; i < size; i++)
    {
        if (data[i] < 0)
        {
            out_data[i] = (exp(data[i]) - 1) * param->alpha;
        }
        else
        {
            out_data[i] = data[i];
        }
    }
    return 0;
}

int ref_elu_uint8(struct tensor* input_tensor, struct tensor* output_tensor, int size, p_elu_param param)
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

    float* data = (float*)sys_malloc(input_size * sizeof(float));
    float* out_data = (float*)sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        data[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    for (int i = 0; i < size; i++)
    {
        if (data[i] < 0)
        {
            out_data[i] = (exp(data[i]) - 1) * param->alpha;
        }
        else
        {
            out_data[i] = data[i];
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

    sys_free(data);
    sys_free(out_data);

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* graph = node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct elu_param* param = (struct elu_param*)node->op.param_mem;

    int elem_num = input_tensor->elem_num;
    void* in_data = input_tensor->data;
    void* out_data = output_tensor->data;

    float scale = 1.f;
    int zero_point = 0;

    _elu_param op_param;
    op_param.alpha = param->alpha;
    op_param.scale = scale;
    op_param.zero_point = zero_point;

    if (input_tensor->data_type == TENGINE_DT_FP32)
        ref_elu_fp32((float*)in_data, (float*)out_data, elem_num, &op_param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ref_elu_uint8(input_tensor, output_tensor, elem_num, &op_param);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_elu_ref_op()
{
    return register_builtin_node_ops(OP_ELU, &hcl_node_ops);
}

int unregister_elu_ref_op()
{
    return unregister_builtin_node_ops(OP_ELU, &hcl_node_ops);
}
