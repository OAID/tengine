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
 * update：qtang@openailab.com
 */

#include "convolution_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include "conv_kernel_ref.h"

// add conv op by wangxinwei for debug conv
//======================================================================================================//

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* bias_tensor = NULL;
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    if (ir_node->input_num > 2)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    }

    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;

    int ret = 0;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_conv_fp32(input_tensor, output_tensor, weight_tensor, bias_tensor, conv_param);
    else if (input_tensor->data_type == TENGINE_DT_FP16)
        ret = ref_conv_fp16(input_tensor, output_tensor, weight_tensor, bias_tensor, conv_param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_conv_uint8(input_tensor, output_tensor, weight_tensor, bias_tensor, conv_param);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_conv_int8(input_tensor, output_tensor, weight_tensor, bias_tensor, conv_param);
    else
    {
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);
        return -1;
    }

    return ret;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;

    /* dynamic get the shape of output tensor */
    int n = input_tensor->dims[0];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int ret = 0;

    if (conv_param->kernel_w == 0)
    {
        conv_param->kernel_w = 1;
        conv_param->pad_w0 = 0;
        conv_param->pad_w1 = 0;
    }
    if (conv_param->kernel_h == 0)
        conv_param->kernel_h = 1;
    if (conv_param->stride_w == 0)
        conv_param->stride_w = 1;
    if (conv_param->stride_h == 0)
        conv_param->stride_h = 1;

    int out_c = conv_param->output_channel;
    int out_h, out_w;

    /* handle the same padding case, which pad_h0 and pad_h1 is -1 (SAME_UPPER)
        -2 (SAME_LOWER) */

    if (conv_param->pad_h0 < 0)
    {
        out_h = (h - 1) / conv_param->stride_h + 1;

        int total_len = (out_h - 1) * conv_param->stride_h + conv_param->kernel_h;
        int pad_num = total_len - h;

        if (conv_param->pad_h0 == -1)
        {
            conv_param->pad_h0 = pad_num / 2;
            conv_param->pad_h1 = pad_num - pad_num / 2;
        }
        else
        {
            conv_param->pad_h1 = pad_num / 2;
            conv_param->pad_h0 = pad_num - pad_num / 2;
        }
    }
    else
    {
        out_h = (h - conv_param->dilation_h * (conv_param->kernel_h - 1) - 1 + conv_param->pad_h0 + conv_param->pad_h1) / conv_param->stride_h + 1;
    }

    if (conv_param->pad_w0 < 0)
    {
        out_w = (w - 1) / conv_param->stride_w + 1;

        int total_len = (out_w - 1) * conv_param->stride_w + conv_param->kernel_w;
        int pad_num = total_len - w;

        if (conv_param->pad_w0 == -1)
        {
            conv_param->pad_w0 = pad_num / 2;
            conv_param->pad_w1 = pad_num - pad_num / 2;
        }
        else
        {
            conv_param->pad_w1 = pad_num / 2;
            conv_param->pad_w0 = pad_num - pad_num / 2;
        }
    }
    else
    {
        out_w = (w - conv_param->dilation_w * (conv_param->kernel_w - 1) - 1 + conv_param->pad_w0 + conv_param->pad_w1) / conv_param->stride_w + 1;
    }

    int dims[4];
    dims[0] = n;

    if (output_tensor->dims[1] != out_c || output_tensor->dims[2] != out_h || output_tensor->dims[3] != out_w)
    {
        dims[1] = out_c;
        dims[2] = out_h;
        dims[3] = out_w;

        for (int i = 0; i < 4; i++)
        {
            if (dims[i] == 0)
                dims[i] = 1;
        }

        ret = set_ir_tensor_shape(output_tensor, dims, 4);
    }

    return ret;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
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
};

int register_conv_ref_op()
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

int unregister_conv_ref_op()
{
    unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
    return 0;
}
