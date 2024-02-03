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

#include "arm_math.h"
#include "sys_port.h"
#include "module.h"
#include "tengine_errno.h"
#include "tengine_log.h"
#include "tengine_ir.h"
#include "cpu_node_ops.h"
#include "tengine_op.h"
#include "pooling_param.h"

void arm_maxpool_q7_HWC_nonsquare(q7_t* Im_in, const uint16_t dim_im_in_x, const uint16_t dim_im_in_y,
                                  const uint16_t ch_im_in, const uint16_t dim_kernel, const uint16_t padding,
                                  const uint16_t stride, const uint16_t dim_im_out_x, const uint16_t dim_im_out_y,
                                  q7_t* bufferA, q7_t* Im_out);

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    struct pool_param* pool_param = (struct pool_param*)ir_node->op.param_mem;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    arm_maxpool_q7_HWC_nonsquare(input_tensor->data, input_tensor->dims[2], input_tensor->dims[1],
                                 input_tensor->dims[3], pool_param->kernel_h, pool_param->pad_h0, pool_param->stride_h,
                                 output_tensor->dims[2], output_tensor->dims[1], NULL, output_tensor->data);

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return -1;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops cmsis_node_ops = {.prerun = NULL,
                                         .run = run,
                                         .reshape = reshape,
                                         .postrun = NULL,
                                         .init_node = NULL,
                                         .release_node = NULL,
                                         .score = score,
                                         .is_ref_op = false};

int register_pooling_cmsis_op()
{
    return register_builtin_node_ops(OP_POOL, &cmsis_node_ops);
}

int unregister_pooling_cmsis_op()
{
    return unregister_builtin_node_ops(OP_POOL, &cmsis_node_ops);
}

AUTO_REGISTER_OPS(reg_pooling_cmsis_ops);
AUTO_UNREGISTER_OPS(unreg_pooling_cmsis_ops);
