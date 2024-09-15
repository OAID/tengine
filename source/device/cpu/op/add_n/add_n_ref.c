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
 * Author: xlchen@openailab.com
 */

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>

struct add_n_op_param
{
    int in_num;
    void** input_data;
};

static int ref_add_n_fp32(const float** input, float* output, int size, const struct add_n_op_param* param)
{
    int in_num = param->in_num;
    for (int i = 0; i < size; ++i)
    {
        output[i] = input[0][i];
        for (int n = 1; n < in_num; n++)
        {
            output[i] += input[n][i];
        }
    }
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct add_n_op_param* add_n_op_param = (struct add_n_op_param*)sys_malloc(sizeof(struct add_n_op_param));
    exec_node->ops_priv = add_n_op_param;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct add_n_op_param* add_n_op_param = (struct add_n_op_param*)exec_node->ops_priv;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    int in_num = ir_node->input_num;
    add_n_op_param->in_num = in_num;
    add_n_op_param->input_data = (void**)sys_malloc(sizeof(void*) * in_num);

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor_a = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    uint32_t elem_num = input_tensor_a->elem_num;
    struct add_n_op_param* add_n_op_param = (struct add_n_op_param*)exec_node->ops_priv;
    for (int i = 0; i < add_n_op_param->in_num; i++)
    {
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        void* data = input_tensor->data;
        add_n_op_param->input_data[i] = data;
    }
    const void** input = (const void**)add_n_op_param->input_data;

    float* output = (float*)output_tensor->data;
    for (uint32_t i = 0; i < elem_num; i++)
    {
        output[i] = 0;
    }
    ref_add_n_fp32((const float**)input, output, elem_num, add_n_op_param);
    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct add_n_op_param* add_n_op_param = (struct add_n_op_param*)exec_node->ops_priv;
    sys_free(add_n_op_param->input_data);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    if (input_tensor->data_type != TENGINE_DT_FP32 || input_tensor->layout != TENGINE_LAYOUT_NCHW)
        return 0;

    return OPS_SCORE_CANDO;
}

static struct node_ops add_n_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = NULL,
    .postrun = postrun,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_add_n_ref_op()
{
    return register_builtin_node_ops(OP_ADD_N, &add_n_node_ops);
}

int unregister_add_n_ref_op()
{
    return unregister_builtin_node_ops(OP_ADD_N, &add_n_node_ops);
}
