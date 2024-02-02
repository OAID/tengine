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
 * Author: bzhang@openailab.com
 */

#include "l2pool_param.h"

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
#include <string.h>

struct ref_l2pool_param
{
    int inc;
    int inh;
    int inw;
    int outh;
    int outw;
    int outc;
    int k_h;
    int k_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int inn;
    float scale[2];    // scale[0]: input scale, scale[1]: output scale
    int zero_point[2]; // zero_point[0]: input zero_point, zero_point[1]: output zero_point
};
#define L2POOL_MAX(a, b) ((a) < (b) ? (b) : (a))
#define L2POOL_MIN(a, b) ((b) < (a) ? (b) : (a))

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ref_l2pool_param* l2pool_op_param = (struct ref_l2pool_param*)sys_malloc(sizeof(struct ref_l2pool_param));
    memset(l2pool_op_param, 0, sizeof(struct ref_l2pool_param));
    exec_node->ops_priv = l2pool_op_param;
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}
void run_l2pool(float* data, float* out_data, struct ref_l2pool_param* param)
{
    for (int c = 0; c < param->inc; c++)
    {
        for (int ph = 0; ph < param->outh; ph++)
        {
            for (int pw = 0; pw < param->outw; pw++)
            {
                // int index = inc * (ph * outw + pw) + c;
                int index = param->inc * (ph * param->outw + pw) + c;
                int h_start = ph * param->stride_h - param->pad_h;
                int h_end = L2POOL_MIN(h_start + param->k_h, param->inh + param->pad_h);
                int w_start = pw * param->stride_w - param->pad_w;
                int w_end = L2POOL_MIN(w_start + param->k_w, param->inw + param->pad_w);
                h_start = L2POOL_MAX(0, ph * param->stride_h - param->pad_h);
                w_start = L2POOL_MAX(0, pw * param->stride_w - param->pad_w);
                h_end = L2POOL_MIN(h_end, param->inh);
                w_end = L2POOL_MIN(w_end, param->inw);
                int pool_size = 0;

                float tmp = 0.0f;
                float val = 0.0f;
                for (int h = h_start; h < h_end; h++)
                {
                    for (int w = w_start; w < w_end; w++)
                    {
                        // val = data[i*param->inh*param->inc * param->inw +h * param->inc * param->inw + w * param->inc
                        // +c];
                        val = data[h * param->inc * param->inw + w * param->inc + c];
                        tmp += val * val;
                        pool_size++;
                    }
                }
                if (tmp == 0)
                {
                    out_data[index] = 0;
                }
                else
                {
                    out_data[index] = sqrt(tmp / pool_size);
                }
            }
        }
    }
}

int ref_l2pool_fp32(float* data, float* out_data, struct ref_l2pool_param* param)
{
    int input_size = param->inc * param->inh * param->inw;
    int output_size = param->outh * param->outw * param->outc;
    for (int i = 0; i < param->inn; i++)
    {
        run_l2pool(data + i * input_size, out_data + i * output_size, param);
    }
    return 0;
}

void ConvertPaddingStyleToParameters(int stride_h, int stride_w,
                                     int in_height, int in_width, int filter_height, int filter_width, int paddingtype,
                                     int out_height, int out_width,
                                     int* padding_width, int* padding_height)
{
    if (paddingtype == 0 || paddingtype == 2)
    {
        *padding_width = 0;
        *padding_height = 0;
    }
    else if (paddingtype == 1)
    {
        *padding_width = (int)(((out_width - 1) * stride_w + filter_width - in_width) / 2);
        *padding_height = (int)(((out_height - 1) * stride_h + filter_height - in_height) / 2);
    }

    return;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
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

    struct ref_l2pool_param* op_param = (struct ref_l2pool_param*)exec_node->ops_priv;
    struct l2pool_param* l2pool_param_op = (struct l2pool_param*)ir_node->op.param_mem;

    int input_c = input_tensor->dims[1];
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];
    int input_n = input_tensor->dims[0];
    int output_h = output_tensor->dims[2];
    int output_w = output_tensor->dims[3];
    int output_c = output_tensor->dims[1];
    int padding_w = 0;
    int padding_h = 0;

    ConvertPaddingStyleToParameters(l2pool_param_op->stride_h, l2pool_param_op->stride_w, input_h, input_w,
                                    l2pool_param_op->kernel_h, l2pool_param_op->kernel_w, l2pool_param_op->paddingType,
                                    output_h, output_w, &padding_w, &padding_h);

    op_param->inc = input_c;
    op_param->inh = input_h;
    op_param->inw = input_w;
    op_param->inn = input_n;
    op_param->k_h = l2pool_param_op->kernel_h;
    op_param->k_w = l2pool_param_op->kernel_w;
    op_param->outh = output_h;
    op_param->outw = output_w;
    op_param->pad_h = padding_h;
    op_param->pad_w = padding_w;
    op_param->stride_h = l2pool_param_op->stride_h;
    op_param->stride_w = l2pool_param_op->stride_w;

    ref_l2pool_fp32((float*)input_tensor->data, (float*)output_tensor->data, op_param);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_l2pool_ref_op()
{
    return register_builtin_node_ops(OP_L2POOL, &hcl_node_ops);
}

int unregister_l2pool_ref_op()
{
    return unregister_builtin_node_ops(OP_L2POOL, &hcl_node_ops);
}
