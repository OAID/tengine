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
 * Author: haitao@openailab.com
 */

#include "softmax_param.h"

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

#include "softmax_kernel_ref.h"

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

    struct softmax_param* softmax_param = (struct softmax_param*)ir_node->op.param_mem;

    // Check: axis must be in the range: [-input_tensor->dim_num, input_tensor->dim_num)
    // Note: Here we always assume 0 <= input_tensor->dim_num
    int axis = softmax_param->axis;
    if (axis < -input_tensor->dim_num || input_tensor->dim_num <= axis)
    {
        TLOG_ERR("Input softmax axis %d not to be supported.\n", axis);
        return -1;
    }
    // In case axis is negative (and is not last dimension -1), add input tensor's dimension
    axis += input_tensor->dim_num;
    axis %= input_tensor->dim_num;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
    {
        ret = ref_softmax_fp32(input_tensor, output_tensor, axis);
    }
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
    {
        ret = ref_softmax_uint8(input_tensor, output_tensor, axis);
    }
    else if (input_tensor->data_type == TENGINE_DT_INT8)
    {
        ret = ref_softmax_int8(input_tensor, output_tensor, axis);
    }
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

    return ret;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    int ret = 0;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (input_tensor->dims[0] != output_tensor->dims[0] || input_tensor->dims[1] != output_tensor->dims[1] || input_tensor->dims[2] != output_tensor->dims[2] || input_tensor->dims[3] != output_tensor->dims[3])
        ret = set_ir_tensor_shape(output_tensor, input_tensor->dims, input_tensor->dim_num);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_softmax_ref_op()
{
    return register_builtin_node_ops(OP_SOFTMAX, &hcl_node_ops);
}

int unregister_softmax_ref_op()
{
    return unregister_builtin_node_ops(OP_SOFTMAX, &hcl_node_ops);
}
