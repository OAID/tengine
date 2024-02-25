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

#include "convolution_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"

#include "arm_math.h"

struct cmsis_param
{
    uint16_t bias_shift;
    uint16_t out_shift;
};

arm_status arm_convolve_HWC_q7_nonsquare(const q7_t* Im_in, const uint16_t dim_im_in_x, const uint16_t dim_im_in_y,
                                         const uint16_t ch_im_in, const q7_t* wt, const uint16_t ch_im_out,
                                         const uint16_t dim_kernel_x, const uint16_t dim_kernel_y,
                                         const uint16_t padding_x, const uint16_t padding_y, const uint16_t stride_x,
                                         const uint16_t stride_y, const q7_t* bias, const uint16_t bias_shift,
                                         const uint16_t out_shift, q7_t* Im_out, const uint16_t dim_im_out_x,
                                         const uint16_t dim_im_out_y, q15_t* bufferA, q7_t* bufferB);

static inline int cal_shift(int scale)
{
    int shift = 0;

    while ((1 << shift) < scale)
        shift++;

    return shift;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    int bias_shift = 0;
    int out_shift = 0;

    if (ir_node->input_num > 2)
    {
        struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        int scale = ir_tensor->scale;
        bias_shift = cal_shift(scale);
    }

    struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    int scale = ir_tensor->scale;
    out_shift = cal_shift(scale);

    struct cmsis_param* param = (struct cmsis_param*)sys_malloc(sizeof(struct cmsis_param));

    param->bias_shift = bias_shift;
    param->out_shift = out_shift;

    exec_node->ops_priv = param;

    /*2*ch_im_in*dim_kernel*dim_kernel */
    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;
    exec_node->shared_mem_size = sizeof(q15_t) * 2 * conv_param->input_channel * conv_param->kernel_h * conv_param->kernel_w;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct cmsis_param* cmsis_param = (struct cmsis_param*)exec_node->ops_priv;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;

    int ret = arm_convolve_HWC_q7_nonsquare(
        input_tensor->data, input_tensor->dims[2], input_tensor->dims[1], input_tensor->dims[3], weight_tensor->data,
        weight_tensor->dims[3], conv_param->kernel_w, conv_param->kernel_h, conv_param->pad_w0, conv_param->pad_h0,
        conv_param->stride_w, conv_param->stride_h, bias_tensor->data, cmsis_param->bias_shift, cmsis_param->out_shift,
        output_tensor->data, output_tensor->dims[2], output_tensor->dims[1], exec_graph->shared_mem, NULL);

    if (ret != ARM_MATH_SUCCESS)
    {
        TLOG_ERR("arm convolve failed\n");
        return -1;
    }

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    /* do not support reshape */
    return -1;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops cmsis_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = reshape,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_conv_cmsis_op()
{
    return register_builtin_node_ops(OP_CONV, &cmsis_node_ops);
}

int unregister_conv_cmsis_op()
{
    unregister_builtin_node_ops(OP_CONV, &cmsis_node_ops);
    return 0;
}
