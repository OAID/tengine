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

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"
#include "utility/float.h"
#include "utility/sys_port.h"
#include "utility/log.h"

#include <math.h>

#if defined(__APPLE__) || defined(_MSC_VER)
#include <stdio.h>
#endif

int ref_softplus_fp32(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread)
{
    int w = input_tensor->dims[3];
    int h = output_tensor->dims[2];
    int channels = input_tensor->dims[1];
    int size = h * w;
    int c_step = h * w;

    float* input_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

#pragma omp parallel for num_threads(num_thread)
    for (int q = 0; q < channels; q++)
    {
        float* src = input_data + c_step * q;
        float* dst = out_data + c_step * q;

        for (int i = 0; i < size; i++)
        {
            dst[i] = log(exp(src[i]) + 1.0f);
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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_softplus_fp32(input_tensor, output_tensor, exec_graph->num_thread);
    else
        printf("Input data type %d not to be supported.\n", input_tensor->data_type);

    return ret;
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
    .is_ref_op = true};

int register_softplus_ref_op()
{
    return register_builtin_node_ops(OP_SOFTPLUS, &hcl_node_ops);
}

int unregister_softplus_ref_op()
{
    return unregister_builtin_node_ops(OP_SOFTPLUS, &hcl_node_ops);
}
