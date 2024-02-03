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
 * Author: qli@openailab.com
 */

#include "rnn_param.h"

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

struct rnn_ref_param
{
    float* init_h_data;
    float* bias;
    float* kernel;

    int seq_lens;
    int batch_size;
    int input_size;
    int output_len;
    int hidden_size;
};
static void concat_axis_1_rnn(const float* a, const float* b, float* c, int m, int n1, int n2)
{
    int n = n1 + n2;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            c[j + i * n] = a[j + i * n1];
        }
        for (int j = 0; j < n2; j++)
        {
            c[j + i * n + n1] = b[j + i * n2];
        }
    }
}

static void do_gemm_rnn(const float* a, const float* b, float* c, int m, int k, int n, int lda, int ldb, int ldc)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            c[i * n + j] = 0.f;
            for (int p = 0; p < k; p++)
            {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
}

static int do_RNN_step(const float* input, float* init_h, const float* kernel, const float* bias, int batch_size,
                       int input_size, int hidden_size)
{
    int input_total_size = input_size + hidden_size;
    int batch_cell_size = hidden_size * batch_size;

    float* ig = (float*)malloc(batch_cell_size * sizeof(float));

    float* merged_input = (float*)malloc(sizeof(float) * batch_size * (input_total_size));
    float* matmul_result = (float*)malloc(sizeof(float) * batch_size * hidden_size);

    // merge input
    concat_axis_1_rnn(input, init_h, merged_input, batch_size, input_size, hidden_size);

    // do gemm
    do_gemm_rnn(merged_input, kernel, matmul_result, batch_size, input_total_size, hidden_size, input_total_size,
                hidden_size, hidden_size);

    // add bias
    if (bias)
    {
        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < hidden_size; j++)
                matmul_result[i * hidden_size + j] += bias[j];
    }
    // activation
    for (int i = 0; i < batch_cell_size; i++)
    {
        ig[i] = tanh(matmul_result[i]);
        init_h[i] = ig[i];
    }

    // free memory
    free(merged_input);
    free(matmul_result);
    free(ig);

    return 0;
}

static int ref_rnn_fp32(float* input, float* output, struct rnn_ref_param* param)
{
    float* init_h = (float*)malloc((unsigned long)param->batch_size * param->hidden_size * sizeof(float));
    if (param->init_h_data)
    {
        for (int i = 0; i < param->batch_size; i++)
        {
            memcpy(init_h + i * param->hidden_size, param->init_h_data, param->hidden_size * sizeof(float));
        }
    }
    else
    {
        memset(init_h, 0x0, sizeof((unsigned long)param->batch_size * param->hidden_size * sizeof(float)));
    }

    int ret = 0;
    for (int i = 0; i < param->seq_lens; i++)
    {
        const float* seq_input = input + i * param->batch_size * param->input_size;

        if (!do_RNN_step(seq_input, init_h, param->kernel, param->bias, param->batch_size, param->input_size,
                         param->hidden_size))
        {
            ret = -1;
            break;
        }
        // outputs [batch_size,seq_len,hidden_size]
        // final_state [batch_size,hidden_size]
        if (i + param->output_len >= param->seq_lens)
        {
            memcpy(output, init_h, (unsigned long)param->batch_size * param->hidden_size * sizeof(float));
            output += param->batch_size * param->hidden_size;
        }
    }
    free(init_h);
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

static struct tensor* bias_tensor;
static float* init_h_data;

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    int in_num = ir_node->input_num;
    struct rnn_param* rnn_param = (struct rnn_param*)ir_node->op.param_mem;
    struct tensor* init_h_tensor;

    for (int count = 0; count < in_num; count++)
    {
        input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[count]);
        if (strstr(input_tensor->name, rnn_param->inithiddenname) != NULL)
        {
            init_h_tensor = input_tensor;
        }
        if (strstr(input_tensor->name, rnn_param->biasname) != NULL)
        {
            bias_tensor = input_tensor;
        }
    }

    if (init_h_tensor)
    {
        init_h_data = (float*)init_h_tensor->data;
    }
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* kernel_tensor;
    struct tensor* output_tensor;
    struct rnn_ref_param rnn_ref_param;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    kernel_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct rnn_param* rnn_param = (struct rnn_param*)ir_node->op.param_mem;

    int input_size = rnn_param->input_size;
    int hidden_size = rnn_param->hidden_size;

    float* output = (float*)output_tensor->data;
    float* input = (float*)input_tensor->data;

    int seq_lens = input_tensor->dims[0];
    int batch_size = input_tensor->dims[1];
    int output_len = rnn_param->output_len;

    float* init_h = (float*)malloc((size_t)batch_size * hidden_size * sizeof(float));
    if (init_h == NULL)
    {
        return -1;
    }
    if (init_h_data)
    {
        for (int i = 0; i < batch_size; i++)
        {
            memcpy(init_h + i * hidden_size, init_h_data, hidden_size * sizeof(float));
        }
    }
    else
    {
        memset(init_h, 0x0, sizeof(batch_size * hidden_size * sizeof(float)));
    }
    float* kernel = (float*)kernel_tensor->data;
    float* bias = NULL;

    if (bias_tensor)
        bias = (float*)bias_tensor->data;

    rnn_ref_param.init_h_data = init_h_data;
    rnn_ref_param.bias = bias;
    rnn_ref_param.kernel = kernel;
    rnn_ref_param.seq_lens = seq_lens;
    rnn_ref_param.batch_size = batch_size;
    rnn_ref_param.input_size = input_size;
    rnn_ref_param.output_len = output_len;
    rnn_ref_param.hidden_size = hidden_size;

    if (ref_rnn_fp32(input, output, &rnn_ref_param) < 0)
        return -1;

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

int register_rnn_ref_op()
{
    return register_builtin_node_ops(OP_RNN, &hcl_node_ops);
}

int unregister_rnn_ref_op()
{
    return unregister_builtin_node_ops(OP_RNN, &hcl_node_ops);
}
