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

#include "comparison_param.h"

#include "comparison_kernel_ref.h"

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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* graph = node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* input_tensor1 = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct comparison_param* param = (struct comparison_param*)node->op.param_mem;

    void* input0 = input_tensor->data;
    void* input1 = input_tensor1->data;
    void* output = output_tensor->data;

    _comparison_param op_param;
    if (input_tensor1->dim_num == 4)
    {
        op_param.shape1[0] = input_tensor1->dims[0];
        op_param.shape1[1] = input_tensor1->dims[1];
        op_param.shape1[2] = input_tensor1->dims[2];
        op_param.shape1[3] = input_tensor1->dims[3];
    }
    else if (input_tensor1->dim_num == 1)
    {
        op_param.shape1[0] = 1;
        op_param.shape1[1] = input_tensor1->dims[0];
        op_param.shape1[2] = 1;
        op_param.shape1[3] = 1;
    }

    if (input_tensor->dim_num == 4)
    {
        op_param.shape0[0] = input_tensor->dims[0];
        op_param.shape0[1] = input_tensor->dims[1];
        op_param.shape0[2] = input_tensor->dims[2];
        op_param.shape0[3] = input_tensor->dims[3];
    }
    else if (input_tensor->dim_num == 1)
    {
        op_param.shape0[0] = 1;
        op_param.shape0[1] = input_tensor->dims[0];
        op_param.shape0[2] = 1;
        op_param.shape0[3] = 1;
    }

    op_param.layout = input_tensor->layout;
    op_param.type = param->type;

    return ref_comparison_fp32((float*)input0, (float*)input1, (float*)output, &op_param);
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

int register_comparison_ref_op()
{
    return register_builtin_node_ops(OP_COMPARISON, &hcl_node_ops);
}

int unregister_comparison_ref_op()
{
    return unregister_builtin_node_ops(OP_COMPARISON, &hcl_node_ops);
}
