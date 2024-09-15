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

#include "logical_param.h"

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

struct logical_param_ref
{
    int type;
    int shape0[4];
    int shape1[4];
};

void logical_and(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
                 float* output)
{
    if (input1_count4 == 1)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) && input1[0];
        }
    }
    else if (input_count4 == input1_count4)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) && (*input1++);
        }
    }
    else if (input_count4 == 1)
    {
        for (int i = 0; i < input1_count4; ++i)
        {
            *output++ = (*input1++) && input0[0];
        }
    }

    else
    {
        return;
    }
    return;
}

void logical_or(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
                float* output)
{
    if (input1_count4 == 1)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) || input1[0];
        }
    }
    else if (input_count4 == input1_count4)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (*input0++) || (*input1++);
        }
    }
    else if (input_count4 == 1)
    {
        for (int i = 0; i < input1_count4; ++i)
        {
            *output++ = (*input1++) || input0[0];
        }
    }

    else
    {
        return;
    }
    return;
}

static int ref_logical_fp32(float* input0, float* input1, float* output, struct logical_param_ref* param,
                            int num_thread)
{
    int input_hw = param->shape0[2] * param->shape0[3];
    int input_hw_1 = param->shape1[2] * param->shape1[3];
    int input_count4 = param->shape0[0] * param->shape0[1] * input_hw;
    int input1_count4 = param->shape1[0] * param->shape1[1] * input_hw_1;

    switch (param->type)
    {
    case 0: // LogicalAnd
    {
        logical_and(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, output);
        break;
    }
    case 1: // LogicalOr
    {
        logical_or(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, output);
        break;
    }
    default:
        return -1;
        ;
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
    struct tensor* input_tensor0;
    struct tensor* input_tensor1;
    struct tensor* output_tensor;
    int layout = ir_graph->graph_layout;

    if (ir_node->input_num != 2)
    {
        TLOG_ERR("logical op need 2 input tensor!\n");
        return -1;
    }

    input_tensor0 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct logical_param* logical_param = (struct logical_param*)ir_node->op.param_mem;
    struct logical_param_ref logical_param_ref;

    logical_param_ref.shape0[0] = 1;
    logical_param_ref.shape0[1] = 1;
    logical_param_ref.shape0[2] = 1;
    logical_param_ref.shape0[3] = 1;

    logical_param_ref.shape1[0] = 1;
    logical_param_ref.shape1[1] = 1;
    logical_param_ref.shape1[2] = 1;
    logical_param_ref.shape1[3] = 1;

    if (input_tensor0->dims[0] != 0)
        logical_param_ref.shape0[0] = input_tensor0->dims[0];
    if (input_tensor0->dims[1] != 0)
        logical_param_ref.shape0[1] = input_tensor0->dims[1];
    if (input_tensor0->dims[2] != 0)
        logical_param_ref.shape0[2] = input_tensor0->dims[2];
    if (input_tensor0->dims[3] != 0)
        logical_param_ref.shape0[3] = input_tensor0->dims[3];

    if (input_tensor1->dims[0] != 0)
        logical_param_ref.shape1[0] = input_tensor1->dims[0];
    if (input_tensor1->dims[1] != 0)
        logical_param_ref.shape1[1] = input_tensor1->dims[1];
    if (input_tensor1->dims[2] != 0)
        logical_param_ref.shape1[2] = input_tensor1->dims[2];
    if (input_tensor1->dims[3] != 0)
        logical_param_ref.shape1[3] = input_tensor1->dims[3];

    logical_param_ref.type = logical_param->type;

    int ret = ref_logical_fp32((float*)input_tensor0->data, (float*)input_tensor1->data,
                               (float*)output_tensor->data, &logical_param_ref, exec_graph->num_thread);
    if (ret != 0)
        return -1;

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

int register_logical_ref_op()
{
    return register_builtin_node_ops(OP_LOGICAL, &hcl_node_ops);
}

int unregister_logical_ref_op()
{
    return unregister_builtin_node_ops(OP_LOGICAL, &hcl_node_ops);
}
