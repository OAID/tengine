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
 * Author: jjzeng@openailab.com
 * Update: hhchen@openailab.com
 */

#include "argmax_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <stdio.h>

struct argmax_op_param
{
    int axis;
    int axis_size;
    int inner_size;
    int outer_size;
    int keepdims;
};

static int ref_argmax_fp32(float* input, int* output, const struct argmax_op_param* param)
{
    float max_value;
    int max_value_index;
    float current;

    int axis_size = param->axis_size;
    int outer_size = param->outer_size;
    int inner_size = param->inner_size;

    for (int outer = 0; outer < outer_size; ++outer)
    {
        for (int inner = 0; inner < inner_size; ++inner)
        {
            max_value = input[outer * axis_size * inner_size + inner];
            max_value_index = 0;
            for (int i = 1; i < axis_size; ++i)
            {
                current = input[(outer * axis_size + i) * inner_size + inner];
                if (current > max_value)
                {
                    max_value = current;
                    max_value_index = i;
                }
            }
            output[outer * inner_size + inner] = max_value_index;
        }
    }

    return 0;
}

static int ref_argmax_uint8(uint8_t* input, uint8_t* output, const struct argmax_op_param* param)
{
    uint8_t max_value;
    int max_value_index;
    uint8_t current;

    int axis_size = param->axis_size;
    int outer_size = param->outer_size;
    int inner_size = param->inner_size;

    for (int outer = 0; outer < outer_size; ++outer)
    {
        for (int inner = 0; inner < inner_size; ++inner)
        {
            max_value = input[outer * axis_size * inner_size + inner];
            max_value_index = 0;
            for (int i = 1; i < axis_size; ++i)
            {
                current = input[(outer * axis_size + i) * inner_size + inner];
                if (current > max_value)
                {
                    max_value = current;
                    max_value_index = i;
                }
            }
            output[outer * inner_size + inner] = max_value_index;
        }
    }

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct argmax_op_param* argmax_op_param = (struct argmax_op_param*)sys_malloc(sizeof(struct argmax_op_param));
    argmax_op_param->axis = 0;
    argmax_op_param->axis_size = 1;
    argmax_op_param->inner_size = 1;
    argmax_op_param->outer_size = 1;
    argmax_op_param->keepdims = 1;
    exec_node->ops_priv = argmax_op_param;
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
    struct tensor* output_tensor;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct argmax_op_param* argmax_op_param = (struct argmax_op_param*)exec_node->ops_priv;
    struct argmax_param* argmax_param = (struct argmax_param*)ir_node->op.param_mem;
    argmax_op_param->axis = argmax_param->axis;
    argmax_op_param->keepdims = argmax_param->keepdims;
    argmax_op_param->axis_size = input_tensor->dims[argmax_param->axis];

    int outer_size = 1;
    for (int i = 0; i < argmax_param->axis; ++i)
    {
        outer_size *= input_tensor->dims[i];
    }

    int inner_size = 1;
    const int dims_count = argmax_param->keepdims;
    for (int i = argmax_param->axis + 1; i < 3; i++)
    {
        inner_size *= input_tensor->dims[i];
    }

    argmax_op_param->inner_size = inner_size;
    argmax_op_param->outer_size = outer_size;

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
    void* in_data = input_tensor->data;
    void* out_data = output_tensor->data;

    struct argmax_op_param* argmax_op_param = (struct argmax_op_param*)exec_node->ops_priv;

    if (input_tensor->data_type == TENGINE_DT_FP32)
        ref_argmax_fp32((float*)in_data, (int*)out_data, argmax_op_param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ref_argmax_uint8((uint8_t*)in_data, (uint8_t*)out_data, argmax_op_param);

    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops argmax_node_ops = {.prerun = prerun,
                                          .run = run,
                                          .reshape = NULL,
                                          .postrun = postrun,
                                          .init_node = init_node,
                                          .release_node = release_node,
                                          .score = score,
                                          .is_ref_op = true};

int register_argmax_ref_op()
{
    return register_builtin_node_ops(OP_ARGMAX, &argmax_node_ops);
}

int unregister_argmax_ref_op()
{
    return unregister_builtin_node_ops(OP_ARGMAX, &argmax_node_ops);
}
