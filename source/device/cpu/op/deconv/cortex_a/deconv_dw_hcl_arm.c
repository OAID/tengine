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
 * Author: haoluo@openailab.com
 */

#include "deconv_param.h"

#include "cortex_a/deconv_dw_kernel_arm.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* bias_tensor = NULL;
    struct tensor* output_tensor = NULL;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct deconv_param* deconv_param = (struct deconv_param*)ir_node->op.param_mem;

    if (deconv_dw_run(input_tensor, weight_tensor, bias_tensor, output_tensor, deconv_param, num_thread, cpu_affinity) < 0)
    {
        TLOG_ERR("hcl deconv run failed\n");
        // set_tengine_errno(EFAULT);
        return -1;
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

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct deconv_param* param = (struct deconv_param*)exec_node->op.param_mem;
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor;
    struct tensor* output_tensor;

    int pads[4];
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int stride_h = param->stride_h;
    int stride_w = param->stride_w;
    int dilation_h = param->dilation_h;
    int dilation_w = param->dilation_w;
    pads[0] = param->pad_h0;
    pads[1] = param->pad_w0;
    pads[2] = param->pad_h1;
    pads[3] = param->pad_w1;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int in_c = input_tensor->dims[1] / group;
    int out_c = output_tensor->dims[1] / group;

    if (param->group > 1 && in_c == 1 && out_c == 1)
        return OPS_SCORE_BEST;
    else
        return 0;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = false};

int register_deconv_dw_hcl_arm_op()
{
    return register_builtin_node_ops(OP_DECONV, &hcl_node_ops);
}

int unregister_deconv_dw_hcl_arm_op()
{
    unregister_builtin_node_ops(OP_DECONV, &hcl_node_ops);
    return 0;
}
