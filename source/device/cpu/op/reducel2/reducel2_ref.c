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
 * Author: bhu@openailab.com
 */

#include "reducel2_param.h"

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

struct ref_reducel2_param
{
    int axis;
    int dims[4];
    float scale[2];
    int zero[2];
};

static int ref_reducel2_fp32(float* in_data, float* out_data, const struct ref_reducel2_param* param)
{
    int in_size = 1;
    int out_size = 1;

    for (int i = 0; i < param->axis; i++)
    {
        out_size = out_size * param->dims[i];
    }
    for (int i = param->axis; i < 4; i++)
    {
        in_size = in_size * param->dims[i];
    }

    for (int i = 0; i < out_size; i++)
    {
        float* data = in_data + i * in_size;
        float sum = 0;
        for (int j = 0; j < in_size; j++)
        {
            sum += data[j] * data[j];
        }

        out_data[i] = sqrt(sum);
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
    struct reducel2_param* op_param = (struct reducel2_param*)ir_node->op.param_mem;

    void* in_data = (void*)input_tensor->data;
    void* out_data = (void*)output_tensor->data;

    struct ref_reducel2_param param;

    if (op_param->axis < 0)
        param.axis = op_param->axis + input_tensor->dim_num;
    else
        param.axis = op_param->axis;

    for (unsigned int i = 0; i < +input_tensor->dim_num; i++)
    {
        param.dims[i] = +input_tensor->dims[i];
    }
    for (unsigned int i = +input_tensor->dim_num; i < 4; i++)
    {
        param.dims[i] = 1;
    }
    int ret = ref_reducel2_fp32((float*)in_data, (float*)out_data, &param);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops reducel2_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_reducel2_ref_op()
{
    return register_builtin_node_ops(OP_REDUCEL2, &reducel2_node_ops);
}

int unregister_reducel2_ref_op()
{
    return unregister_builtin_node_ops(OP_REDUCEL2, &reducel2_node_ops);
}
