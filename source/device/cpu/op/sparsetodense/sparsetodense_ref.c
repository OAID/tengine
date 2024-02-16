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

#include "sparsetodense_param.h"

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

int ref_sparsetodense_fp32(struct tensor* input_tensor, struct tensor* output_shape_tensor,
                           struct tensor* sparse_values_tensor, struct tensor* output_tensor,
                           struct sparsetodense_param* param, int num_thread)
{
    int output_dim_size = output_shape_tensor->dim_num;
    int indices_dim_size = input_tensor->dim_num;
    int sparse_value_size = sparse_values_tensor->dim_num;
    float default_value = param->default_value;

    int* input = (int*)input_tensor->data;
    int* outout_shape = (int*)output_shape_tensor->data;
    int* sparse_values = (int*)sparse_values_tensor->data;
    float* output = (float*)output_tensor->data;

    if (output_dim_size == 1)
    {
        for (int i = 0; i < outout_shape[0]; i++)
        {
            output[i] = default_value;
        }

        if (sparse_value_size == 0)
        {
            if (indices_dim_size == 0)
            {
                output[*input] = *sparse_values;
            }

            else if (indices_dim_size == 1)
            {
                for (int i = 0; i < input_tensor->dims[0]; i++)
                {
                    output[input[i]] = *sparse_values;
                }
            }

            else
            {
                return -1;
            }
        }

        else if (sparse_value_size == 1)
        {
            if (indices_dim_size == 0)
            {
                output[*input] = sparse_values[0];
            }

            else if (indices_dim_size == 1)
            {
                for (int i = 0; i < input_tensor->dims[0]; i++)
                {
                    output[input[i]] = sparse_values[i];
                }
            }

            else
            {
                return -1;
            }
        }
    }

    if (output_dim_size == 2)
    {
        for (int i = 0; i < outout_shape[0] * outout_shape[1]; i++)
        {
            output[i] = default_value;
        }

        if (indices_dim_size != 2)
        {
            return -1;
        }

        if (sparse_value_size == 0)
        {
            for (int i = 0; i < input_tensor->dims[0] * 2; i += 2)
            {
                int x = input[i];
                int y = input[i + 1];
                output[outout_shape[1] * x + y] = *sparse_values;
            }
        }
        else if (sparse_value_size == 1)
        {
            for (int i = 0; i < input_tensor->dims[0] * 2; i += 2)
            {
                int x = input[i];
                int y = input[i + 1];
                output[outout_shape[1] * x + y] = sparse_values[i / 2];
            }
        }
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
    struct tensor* output_shape_tensor;
    struct tensor* sparse_values_tensor;
    struct tensor* output_tensor;
    int layout = ir_graph->graph_layout;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_shape_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    sparse_values_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct sparsetodense_param* sparsetodense_param = (struct sparsetodense_param*)ir_node->op.param_mem;

    int ret = ref_sparsetodense_fp32(input_tensor, output_shape_tensor, sparse_values_tensor, output_tensor,
                                     sparsetodense_param, exec_graph->num_thread);
    if (ret != 0)
        return -1;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_sparsetodense_ref_op()
{
    return register_builtin_node_ops(OP_SPARSETODENSE, &hcl_node_ops);
}

int unregister_sparsetodense_ref_op()
{
    return unregister_builtin_node_ops(OP_SPARSETODENSE, &hcl_node_ops);
}
