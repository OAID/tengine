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
#include "expand_param.h"
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
#include <string.h>
#include <limits.h>

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

int ref_expand_fp32(float* in1_data, float* in2_data, float* out_data, int* in1_dims, int* in2_dims)
{
    int i_n = in1_dims[0] == 0 ? 1 : in1_dims[0];
    int i_c = in1_dims[1] == 0 ? 1 : in1_dims[1];
    int i_h = in1_dims[2] == 0 ? 1 : in1_dims[2];
    int i_w = in1_dims[3] == 0 ? 1 : in1_dims[3];

    int o_n = in2_dims[0] == 0 ? 1 : in2_dims[0];
    int o_c = in2_dims[1] == 0 ? 1 : in2_dims[1];
    int o_h = in2_dims[2] == 0 ? 1 : in2_dims[2];
    int o_w = in2_dims[3] == 0 ? 1 : in2_dims[3];

    int int_max = INT_MAX;
    if (i_n > int_max / i_c || i_h > int_max / (i_n * i_c) || i_w > int_max / (i_n * i_c * i_h))
    {
        TLOG_INFO("input dims overflow!");
        return -1;
    }
    if (o_n > int_max / o_c || o_h > int_max / (o_n * o_c) || o_w > int_max / (o_n * o_c * o_h))
    {
        TLOG_INFO("output dims overflow!");
        return -1;
    }

    int index = 0;
    int i_index = 0;
    if (1 == i_n && 1 == i_h && 1 == i_w && 1 == o_n && i_c == o_c)
    {
        for (int n = 0; n < o_n; ++n)
        {
            for (int c = 0; c < o_c; c++)
            {
                for (int i = 0; i < o_h * o_w; i++)
                {
                    out_data[index++] = in1_data[i_index];
                }
                i_index++;
            }
        }
    }
    else
    {
        int i_size = i_n * i_c * i_h * i_w;
        int refreshed = 0;
        for (int n = 0; n < o_n; n++)
        {
            for (int c = 0; c < o_c; c++)
            {
                for (int h = 0; h < o_h; ++h)
                {
                    for (int w = 0; w < o_w; ++w)
                    {
                        refreshed = 0;
                        if (i_index == i_size)
                            i_index = 0;
                        out_data[index++] = in1_data[i_index];
                        if (i_w != 1)
                        {
                            i_index++;
                            refreshed = 1;
                        }
                    }
                    if (i_h != 1 && refreshed == 0)
                    {
                        i_index++;
                        refreshed = 1;
                    }
                }
                if (i_c != 1 && refreshed == 0)
                {
                    i_index++;
                    refreshed = 1;
                }
            }
            if (i_n != 1 && refreshed == 0)
            {
                i_index++;
            }
        }
    }
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input1_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* input2_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct expand_param* param = (struct expand_param*)ir_node->op.param_mem;

    int dim1_size = input1_tensor->dim_num;
    int dim2_size = input2_tensor->dim_num;

    int* input1_dims = (int*)malloc(sizeof(int) * 4);
    int* input2_dims = (int*)malloc(sizeof(int) * 4);
    for (int i = 0; i < 4; i++)
    {
        input1_dims[i] = 0;
    }

    for (int i = 0; i < 4; i++)
    {
        input2_dims[i] = 0;
    }

    for (int i = 0; i < dim1_size; i++)
    {
        input1_dims[i] = input1_tensor->dims[i];
    }
    for (int i = 0; i < param->dim_num; i++)
    {
        input2_dims[i] = param->ex_shape[i];
    }
    ref_expand_fp32((float*)input1_tensor->data, (float*)input2_tensor->data,
                    (float*)output_tensor->data, input1_dims, input2_dims);

    free(input1_dims);
    free(input2_dims);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops expand_node_ops = {.prerun = NULL,
                                          .run = run,
                                          .reshape = NULL,
                                          .postrun = NULL,
                                          .init_node = init_node,
                                          .release_node = release_node,
                                          .score = score,
                                          .is_ref_op = true};

int register_expand_ref_op()
{
    return register_builtin_node_ops(OP_EXPAND, &expand_node_ops);
}

int unregister_expand_ref_op()
{
    return unregister_builtin_node_ops(OP_EXPAND, &expand_node_ops);
}
