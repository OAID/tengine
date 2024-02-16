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

#include "unsqueeze_param.h"

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

int ref_unsqueeze_fp32(struct tensor* input_tensor, struct tensor* output_tensor)
{
    float* input_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    for (int i = 0; i < input_tensor->elem_num; i++)
    {
        out_data[i] = input_data[i];
    }

    return 0;
}

int ref_unsqueeze_uint8(struct tensor* input_tensor, struct tensor* output_tensor)
{
    uint8_t* input_data = (uint8_t*)input_tensor->data;
    uint8_t* out_data = (uint8_t*)output_tensor->data;

    for (int i = 0; i < input_tensor->elem_num; i++)
    {
        out_data[i] = input_data[i];
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

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_unsqueeze_fp32(input_tensor, output_tensor);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_unsqueeze_uint8(input_tensor, output_tensor);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops unsqueeze_node_ops = {.prerun = NULL,
                                             .run = run,
                                             .reshape = NULL,
                                             .postrun = NULL,
                                             .init_node = init_node,
                                             .release_node = release_node,
                                             .score = score,
                                             .is_ref_op = true};

int register_unsqueeze_ref_op()
{
    return register_builtin_node_ops(OP_UNSQUEEZE, &unsqueeze_node_ops);
}

int unregister_unsqueeze_ref_op()
{
    return unregister_builtin_node_ops(OP_UNSQUEEZE, &unsqueeze_node_ops);
}
