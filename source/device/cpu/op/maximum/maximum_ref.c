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
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>


struct maximum_op_param
{
    int in_num;
    void** input_data;
};

static int ref_maximum_fp32(const float** in_data, float* out_data, int size, const struct maximum_op_param* param)
{
    int in_num = param->in_num;
    for (int i = 0; i < size; i++)
    {
        float max = in_data[0][i];
        for (int n = 1; n < in_num; n++)
        {
            const float* data = in_data[n];
            if (data[i] > max)
                max = data[i];
        }
        out_data[i] = max;
    }
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct maximum_op_param* maximum_op_param = ( struct maximum_op_param* )sys_malloc(sizeof(struct maximum_op_param));
    exec_node->ops_priv = maximum_op_param;

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
    struct maximum_op_param* maximum_op_param = ( struct maximum_op_param* )exec_node->ops_priv;

    int in_num = ir_node->input_num;

    maximum_op_param->in_num = in_num;
    maximum_op_param->input_data = ( void* )sys_malloc(sizeof(void*) * in_num);
    /*for (int i = 0; i < in_num; i++)
    {
        maximum_op_param->input_data[i] = (void *)sys_malloc(sizeof(void));
    }*/

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor_a = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    uint32_t elem_num = input_tensor_a->elem_num;
    struct maximum_op_param* maximum_op_param = ( struct maximum_op_param* )exec_node->ops_priv;
    for (int i = 0; i < maximum_op_param->in_num; i++)
    {
        struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
        void* data = input_tensor->data;
        maximum_op_param->input_data[i] = data;
    }

    const void** input = ( const void** )maximum_op_param->input_data;
    float* output = output_tensor->data;

    ref_maximum_fp32(( const float** )input, output, elem_num, maximum_op_param);

    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct maximum_op_param* maximum_op_param = ( struct maximum_op_param* )exec_node->ops_priv;

    /*for (int i = 0; i < maximum_op_param->in_num; i++)
    {
        sys_free(maximum_op_param->input_data[i]);
    }*/

    sys_free(maximum_op_param->input_data);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops maximum_node_ops = {.prerun = prerun,
                                           .run = run,
                                           .reshape = NULL,
                                           .postrun = postrun,
                                           .init_node = init_node,
                                           .release_node = release_node,
                                           .score = score};

int register_maximum_ref_op()
{
    return register_builtin_node_ops(OP_MAXIMUM, &maximum_node_ops);
}

int unregister_maximum_ref_op()
{
    return unregister_builtin_node_ops(OP_MAXIMUM, &maximum_node_ops);
}
