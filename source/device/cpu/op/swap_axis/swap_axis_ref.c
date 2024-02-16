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
 * Author: hhchen@openailab.com
 */

#include "swap_axis_param.h"

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

static int ref_swap_axis_common(struct tensor* input_tensor, struct tensor* output_tensor, const int* dims, int element_size)
{
    const float* in_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    for (int i = 0; i < dims[0]; i++)
        for (int j = 0; j < dims[3]; j++)
            for (int p = 0; p < dims[2]; p++)
                for (int q = 0; q < dims[1]; q++)
                {
                    int out_index = i * dims[1] * dims[2] * dims[3] * dims[4] + j * dims[2] * dims[1] * dims[4] + p * dims[1] * dims[4] + q * dims[4];
                    int in_index = i * dims[1] * dims[2] * dims[3] * dims[4] + q * dims[2] * dims[3] * dims[4] + p * dims[3] * dims[4] + j * dims[4];
                    memcpy(out_data + out_index * element_size, in_data + in_index * element_size,
                           (size_t)dims[4] * element_size);
                }
    return 0;
}

static int ref_swap_axis_uint8(struct tensor* input_tensor, struct tensor* output_tensor, const int* dims, int element_size)
{
    const uint8_t* in_data = (uint8_t*)input_tensor->data;
    uint8_t* out_data = (uint8_t*)output_tensor->data;

    for (int i = 0; i < dims[0]; i++)
        for (int j = 0; j < dims[3]; j++)
            for (int p = 0; p < dims[2]; p++)
                for (int q = 0; q < dims[1]; q++)
                {
                    int out_index = i * dims[1] * dims[2] * dims[3] * dims[4] + j * dims[2] * dims[1] * dims[4] + p * dims[1] * dims[4] + q * dims[4];
                    int in_index = i * dims[1] * dims[2] * dims[3] * dims[4] + q * dims[2] * dims[3] * dims[4] + p * dims[3] * dims[4] + j * dims[4];
                    memcpy(out_data + out_index * element_size, in_data + in_index * element_size,
                           (size_t)dims[4] * element_size);
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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct swap_axis_param* _param = (struct swap_axis_param*)(ir_node->op.param_mem);
    int in_size = 1;
    for (int i = 0; i < input_tensor->dim_num; i++)
    {
        in_size *= input_tensor->dims[i];
    }
    int dim0 = _param->dim_0;
    int dim1 = _param->dim_1;
    int dims[5];
    if (dim0 > dim1)
    {
        int tmp = dim0;
        dim0 = dim1;
        dim1 = tmp;
    }

    for (int i = 0; i < 5; i++)
        dims[i] = 1;
    // dim0
    for (int i = 0; i < dim0; i++)
        dims[0] *= input_tensor->dims[i];
    // dim1
    dims[1] = input_tensor->dims[dim0];
    // dim2
    for (int i = dim0 + 1; i < dim1; i++)
        dims[2] *= input_tensor->dims[i];
    // dim3
    dims[3] = input_tensor->dims[dim1];
    // dim4
    for (int i = dim1 + 1; i < in_size; i++)
        dims[4] *= input_tensor->dims[i];

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_swap_axis_common(input_tensor, output_tensor, dims, sizeof(float));
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_swap_axis_uint8(input_tensor, output_tensor, dims, sizeof(uint8_t));

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops swap_axis_node_ops = {.prerun = NULL,
                                             .run = run,
                                             .reshape = NULL,
                                             .postrun = NULL,
                                             .init_node = init_node,
                                             .release_node = release_node,
                                             .score = score,
                                             .is_ref_op = true};

int register_swap_axis_ref_op()
{
    return register_builtin_node_ops(OP_SWAP_AXIS, &swap_axis_node_ops);
}

int unregister_swap_axis_ref_op()
{
    return unregister_builtin_node_ops(OP_SWAP_AXIS, &swap_axis_node_ops);
}
