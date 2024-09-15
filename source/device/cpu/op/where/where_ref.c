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
 * Author: bzhang@openailab.com
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

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}
static int ref_where_fp32(float* condition, float* data_a, float* data_b, float* output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = condition[i] ? data_a[i] : data_b[i];
    }
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

    struct tensor* input_tensor_a = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* input_tensor_b = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);

    int elem_num_condition = input_tensor->elem_num;
    int elem_num_a = input_tensor_a->elem_num;
    int elem_num_b = input_tensor_b->elem_num;

    if (elem_num_condition != elem_num_a || elem_num_condition != elem_num_b)
    {
        TLOG_ERR("Tensor size is not equal\n");
        return -1;
    }

    int ret = ref_where_fp32((float*)input_tensor->data, (float*)input_tensor_a->data,
                             (float*)input_tensor_b->data, (float*)output_tensor->data, elem_num_a);
    if (ret < -1)
    {
        TLOG_ERR("where operator execution error\n");
        return -1;
    }

    return 0;
}
static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    int ret = set_ir_tensor_shape(output, input->dims, input->dim_num);
    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = reshape,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_where_ref_op()
{
    return register_builtin_node_ops(OP_WHERE, &hcl_node_ops);
}

int unregister_where_ref_op()
{
    return unregister_builtin_node_ops(OP_WHERE, &hcl_node_ops);
}
