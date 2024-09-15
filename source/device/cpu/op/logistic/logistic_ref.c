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
 * Author: qli@openailab.com
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

struct logical_param
{
    int out_size;
    float scale[2];    // scale[0]: input scale, scale[1]: output scale
    int zero_point[2]; // zero_point[0]: input zero_point, zero_point[1]: output zero_point
};

static int ref_logistic_fp32(float* input_data, float* output_data, struct logical_param* op_param)
{
    for (int i = 0; i < op_param->out_size; i++)
    {
        /* get max */
        output_data[i] = 1.f / (1.f + exp(-input_data[i]));
    }

    return 0;
}

static int ref_logistic_uint8(uint8_t* input, uint8_t* output, struct logical_param* op_param)
{
    for (int i = 0; i < op_param->out_size; i++)
    {
        /* get max */
        output[i] = (1.f / (1.f + exp(-(input[i] - (double)op_param->zero_point[0]) * op_param->scale[0]))) / op_param->scale[1] + op_param->zero_point[1];
    }

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
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    struct logical_param logical_param;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    logical_param.out_size = input_tensor->elem_num;
    logical_param.scale[0] = input_tensor->scale;
    logical_param.scale[1] = output_tensor->scale;
    logical_param.zero_point[0] = input_tensor->zero_point;
    logical_param.zero_point[1] = output_tensor->zero_point;

    if (input_tensor->data_type == TENGINE_DT_FP32)
        ref_logistic_fp32((float*)input_tensor->data, (float*)output_tensor->data, &logical_param);
    else
        ref_logistic_uint8((uint8_t*)input_tensor->data, (uint8_t*)output_tensor->data, &logical_param);

    return 0;
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

int register_logistic_ref_op()
{
    return register_builtin_node_ops(OP_LOGISTIC, &hcl_node_ops);
}

int unregister_logistic_ref_op()
{
    return unregister_builtin_node_ops(OP_LOGISTIC, &hcl_node_ops);
}
