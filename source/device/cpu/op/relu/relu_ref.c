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

#include "relu_param.h"

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

#include "relu_kernel_ref.h"

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

    struct relu_param* relu_param = (struct relu_param*)ir_node->op.param_mem;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_relu_fp32(input_tensor, output_tensor, relu_param->negative_slope);
    else if (input_tensor->data_type == TENGINE_DT_FP16)
#if MACOS
        TLOG_ERR("FP16 not support mac os");
#else
        ret = ref_relu_fp16(input_tensor, output_tensor, relu_param->negative_slope);
#endif
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_relu_uint8(input_tensor, output_tensor, relu_param->negative_slope);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_relu_int8(input_tensor, output_tensor, relu_param->negative_slope);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

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

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_relu_ref_op()
{
    return register_builtin_node_ops(OP_RELU, &hcl_node_ops);
}

int unregister_relu_ref_op()
{
    return unregister_builtin_node_ops(OP_RELU, &hcl_node_ops);
}
