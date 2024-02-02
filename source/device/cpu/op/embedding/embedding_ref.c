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

#include "embedding_param.h"

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

int ref_embed_fp32(float* in_data, float* out_data, float* weight_data, float* bias_data, int input_dim, int num_output,
                   int size, int bias_term, float scale, int zero_point)
{
    for (int i = 0; i < size; i++)
    {
        int word_index = in_data[i];
        if (word_index < 0)
            word_index = 0;
        if (word_index >= input_dim)
            word_index = input_dim - 1;
        const float* embed = (const float*)weight_data + num_output * word_index;
        for (int z = 0; z < num_output; z++)
        {
            out_data[i * num_output + z] = embed[z];
            if (bias_term)
                out_data[i * num_output + z] += bias_data[z];
        }
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* graph = node->graph;

    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    struct embedding_param* param = (struct embedding_param*)node->op.param_mem;

    struct tensor* weight_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* bias_tensor = NULL;
    if (param->bias_term)
    {
        bias_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);
    }

    return ref_embed_fp32((float*)input->data, (float*)output->data, (float*)weight_tensor->data,
                          bias_tensor ? (float*)bias_tensor->data : NULL, param->input_dim, param->num_output,
                          input->elem_size, param->bias_term, 1.0f, 0.0f);
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

int register_embedding_ref_op()
{
    return register_builtin_node_ops(OP_EMBEDDING, &hcl_node_ops);
}

int unregister_embedding_ref_op()
{
    return unregister_builtin_node_ops(OP_EMBEDDING, &hcl_node_ops);
}
