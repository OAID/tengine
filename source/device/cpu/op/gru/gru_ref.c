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
 * Copyright (c) 2020, OPEN AI LAB
 * Author: xwwang@openailab.com
 */

#include "gru_param.h"

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
#include <string.h>

#if defined(__APPLE__) || defined(_MSC_VER)
#include <stdio.h>
#endif

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

int ref_gru_default_fp32(struct tensor* input_tensor, struct tensor* w, struct tensor* r, struct tensor* output_tensor, struct gru_param* param)
{
    int batch_size = input_tensor->dims[1];
    int size = input_tensor->dims[2];
    int hidden_size = param->hidden_size;

    float* x_data = (float*)input_tensor->data;
    float* w_data = (float*)w->data;
    float* r_data = (float*)r->data;
    float* output_data = (float*)output_tensor->data;

    /* initial_h_data buffers */
    float* initial_h_data = (float*)malloc((unsigned long)hidden_size * batch_size * sizeof(float));
    float* output_h_data = (float*)malloc((unsigned long)hidden_size * batch_size * sizeof(float));
    float* h_0 = (float*)malloc((unsigned long)hidden_size * batch_size * sizeof(float));
    memset(initial_h_data, 0, (unsigned long)hidden_size * batch_size * sizeof(float));
    memset(output_h_data, 0, (unsigned long)hidden_size * batch_size * sizeof(float));
    memset(h_0, 0, (unsigned long)hidden_size * batch_size * sizeof(float));

    float* Z_data = (float*)malloc(hidden_size * sizeof(float));
    float* R_data = (float*)malloc(hidden_size * sizeof(float));
    float* H_data = (float*)malloc(hidden_size * sizeof(float));

    int T = input_tensor->dims[1];

    for (int seq = 0; seq < input_tensor->dims[0]; seq++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int q = 0; q < hidden_size; q++)
            {
                float Z = 0;
                float R = 0;
                float w_H = 0;
                float r_H = 0;
                for (int m = 0; m < size; m++)
                {
                    float x_i = x_data[seq * input_tensor->dims[1] * input_tensor->dims[2] + t * input_tensor->dims[2] + m];
                    Z += x_i * w_data[(hidden_size * 0 + q) * input_tensor->dims[2] + m];
                    R += x_i * w_data[(hidden_size * 1 + q) * input_tensor->dims[2] + m];
                    w_H += x_i * w_data[(hidden_size * 2 + q) * input_tensor->dims[2] + m];
                }

                for (int h = 0; h < hidden_size; h++)
                {
                    if (seq == 0)
                    {
                        float h_i = initial_h_data[t * hidden_size + h];
                        Z += h_i * r_data[(hidden_size * 0 + q) * hidden_size + h];
                        R += h_i * r_data[(hidden_size * 1 + q) * hidden_size + h];
                    }
                    else
                    {
                        float h_i = output_h_data[t * hidden_size + h];
                        Z += h_i * r_data[(hidden_size * 0 + q) * hidden_size + h];
                        R += h_i * r_data[(hidden_size * 1 + q) * hidden_size + h];
                    }
                }

                float r_tmp = 1.f / (1.f + exp(-R));
                for (int k = 0; k < hidden_size; k++)
                {
                    if (seq == 0)
                    {
                        r_H += r_tmp * initial_h_data[t * hidden_size + k] * r_data[(hidden_size * 2 + q) * hidden_size + k];
                    }
                    else
                    {
                        r_H += r_tmp * output_h_data[t * hidden_size + k] * r_data[(hidden_size * 2 + q) * hidden_size + k];
                    }
                }

                Z_data[q] = Z;
                R_data[q] = R;
                H_data[q] = w_H + r_H;
            }

            for (int h = 0; h < hidden_size; h++)
            {
                if (seq == 0)
                {
                    float Z = 1.f / (1.f + exp(-Z_data[h]));
                    float H = tanh(H_data[h]);
                    float out = (1 - Z) * H + Z * h_0[h];
                    output_data[t * hidden_size + h] = out;
                    output_h_data[t * hidden_size + h] = out;
                }
                else
                {
                    float Z = 1.f / (1.f + exp(-Z_data[h]));
                    float H = tanh(H_data[h]);
                    float out = (1 - Z) * H + Z * output_h_data[t * hidden_size + h];
                    output_data[t * hidden_size + h] = out;
                    output_h_data[t * hidden_size + h] = out;
                }
            }
        }
    }

    free(initial_h_data);
    free(output_h_data);
    free(h_0);
    free(Z_data);
    free(R_data);
    free(H_data);

    return 0;
}

int ref_gru_with_bias_fp32(struct tensor* input_tensor, struct tensor* w, struct tensor* r, struct tensor* b, struct tensor* output_tensor, struct gru_param* param)
{
    int batch_size = input_tensor->dims[1];
    int size = input_tensor->dims[2];
    int hidden_size = param->hidden_size;

    float* x_data = (float*)input_tensor->data;
    float* w_data = (float*)w->data;
    float* r_data = (float*)r->data;
    float* b_data = (float*)b->data;
    float* output_data = (float*)output_tensor->data;

    /* initial_h_data buffers */
    float* initial_h_data = (float*)malloc((unsigned long)hidden_size * batch_size * sizeof(float));
    float* output_h_data = (float*)malloc((unsigned long)hidden_size * batch_size * sizeof(float));
    float* h_0 = (float*)malloc((unsigned long)hidden_size * batch_size * sizeof(float));
    memset(initial_h_data, 0, (unsigned long)hidden_size * batch_size * sizeof(float));
    memset(output_h_data, 0, (unsigned long)hidden_size * batch_size * sizeof(float));
    memset(h_0, 0, (unsigned long)hidden_size * batch_size * sizeof(float));

    float* Z_data = (float*)malloc(hidden_size * sizeof(float));
    float* R_data = (float*)malloc(hidden_size * sizeof(float));
    float* H_data = (float*)malloc(hidden_size * sizeof(float));

    int T = input_tensor->dims[1];

    for (int seq = 0; seq < input_tensor->dims[0]; seq++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int q = 0; q < hidden_size; q++)
            {
                float Z = 0;
                float R = 0;
                float w_H = 0;
                float r_H = 0;
                float H = 0;
                for (int m = 0; m < size; m++)
                {
                    float x_i = x_data[seq * input_tensor->dims[1] * input_tensor->dims[2] + t * input_tensor->dims[2] + m];
                    Z += x_i * w_data[(hidden_size * 0 + q) * input_tensor->dims[2] + m];
                    R += x_i * w_data[(hidden_size * 1 + q) * input_tensor->dims[2] + m];
                    w_H += x_i * w_data[(hidden_size * 2 + q) * input_tensor->dims[2] + m];
                }

                Z += b_data[hidden_size * 0 + q];
                R += b_data[hidden_size * 1 + q];
                w_H += b_data[hidden_size * 2 + q];

                for (int h = 0; h < hidden_size; h++)
                {
                    if (seq == 0)
                    {
                        float h_i = initial_h_data[t * hidden_size + h];
                        Z += h_i * r_data[(hidden_size * 0 + q) * hidden_size + h];
                        R += h_i * r_data[(hidden_size * 1 + q) * hidden_size + h];
                    }
                    else
                    {
                        float h_i = output_h_data[t * hidden_size + h];
                        Z += h_i * r_data[(hidden_size * 0 + q) * hidden_size + h];
                        R += h_i * r_data[(hidden_size * 1 + q) * hidden_size + h];
                    }
                }

                Z += b_data[hidden_size * 3 + hidden_size * 0 + q];
                R += b_data[hidden_size * 3 + hidden_size * 1 + q];

                float r_tmp = 1.f / (1.f + exp(-R));
                for (int k = 0; k < hidden_size; k++)
                {
                    if (seq == 0)
                    {
                        r_H += r_tmp * initial_h_data[t * hidden_size + k] * r_data[(hidden_size * 2 + q) * hidden_size + k];
                    }
                    else
                    {
                        r_H += r_tmp * output_h_data[t * hidden_size + k] * r_data[(hidden_size * 2 + q) * hidden_size + k];
                    }
                }
                r_H += b_data[hidden_size * 3 + hidden_size * 2 + q];
                Z_data[q] = Z;
                R_data[q] = R;
                H_data[q] = w_H + r_H;
            }

            for (int h = 0; h < hidden_size; h++)
            {
                if (seq == 0)
                {
                    float Z = 1.f / (1.f + exp(-Z_data[h]));
                    float H = tanh(H_data[h]);
                    float out = (1 - Z) * H + Z * h_0[h];
                    output_data[t * hidden_size + h] = out;
                    output_h_data[t * hidden_size + h] = out;
                }
                else
                {
                    float Z = 1.f / (1.f + exp(-Z_data[h]));
                    float H = tanh(H_data[h]);
                    float out = (1 - Z) * H + Z * output_h_data[t * hidden_size + h];
                    output_data[t * hidden_size + h] = out;
                    output_h_data[t * hidden_size + h] = out;
                }
            }
        }
    }

    free(initial_h_data);
    free(output_h_data);
    free(h_0);
    free(Z_data);
    free(R_data);
    free(H_data);

    return 0;
}

int ref_gru_case1_fp32(struct tensor* input_tensor, struct tensor* w, struct tensor* r, struct tensor* b, struct tensor* output_tensor, struct gru_param* param)
{
    int batch_size = input_tensor->dims[1];
    int hidden_size = param->hidden_size;
    float* x_data = (float*)input_tensor->data;
    float* w_data = (float*)w->data;
    float* r_data = (float*)r->data;
    float* b_data = (float*)b->data;

    /* initial_h_data buffers */
    float* initial_h_data = (float*)malloc((unsigned long)hidden_size * batch_size * sizeof(float));
    float* output_h_data = (float*)malloc((unsigned long)hidden_size * batch_size * sizeof(float));
    float* h_0 = (float*)malloc((unsigned long)hidden_size * batch_size * sizeof(float));
    memset(initial_h_data, 0, (unsigned long)hidden_size * batch_size * sizeof(float));
    memset(output_h_data, 0, (unsigned long)hidden_size * batch_size * sizeof(float));
    memset(h_0, 0, (unsigned long)hidden_size * batch_size * sizeof(float));

    float* Z_data = (float*)malloc(hidden_size * sizeof(float));
    float* R_data = (float*)malloc(hidden_size * sizeof(float));
    float* H_data = (float*)malloc(hidden_size * sizeof(float));

    float* output_data = (float*)output_tensor->data;
    int T = input_tensor->dims[1];
    int size = input_tensor->dims[2];

    for (int seq = 0; seq < input_tensor->dims[0]; seq++)
    {
        for (int t = 0; t < T; t++)
        {
            for (int q = 0; q < hidden_size; q++)
            {
                float Z = 0;
                float R = 0;
                float w_H = 0;
                float r_H = 0;
                for (int m = 0; m < size; m++)
                {
                    float x_i = x_data[seq * input_tensor->dims[1] * input_tensor->dims[2] + t * input_tensor->dims[2] + m];
                    Z += x_i * w_data[(hidden_size * 0 + q) * input_tensor->dims[2] + m];
                    R += x_i * w_data[(hidden_size * 1 + q) * input_tensor->dims[2] + m];
                    w_H += x_i * w_data[(hidden_size * 2 + q) * input_tensor->dims[2] + m];
                }

                Z += b_data[hidden_size * 0 + q];
                R += b_data[hidden_size * 1 + q];
                w_H += b_data[hidden_size * 2 + q];

                for (int h = 0; h < hidden_size; h++)
                {
                    if (seq == 0)
                    {
                        float h_i = initial_h_data[t * hidden_size + h];
                        Z += h_i * r_data[(hidden_size * 0 + q) * hidden_size + h];
                        R += h_i * r_data[(hidden_size * 1 + q) * hidden_size + h];
                        r_H += h_i * r_data[(hidden_size * 2 + q) * hidden_size + h];
                    }
                    else
                    {
                        float h_i = output_h_data[t * hidden_size + h];
                        Z += h_i * r_data[(hidden_size * 0 + q) * hidden_size + h];
                        R += h_i * r_data[(hidden_size * 1 + q) * hidden_size + h];
                        r_H += h_i * r_data[(hidden_size * 2 + q) * hidden_size + h];
                    }
                }

                Z += b_data[hidden_size * 3 + hidden_size * 0 + q];
                R += b_data[hidden_size * 3 + hidden_size * 1 + q];
                r_H += b_data[hidden_size * 3 + hidden_size * 2 + q];

                float r_tmp = 1.f / (1.f + exp(-R));
                Z_data[q] = Z;
                R_data[q] = R;
                H_data[q] = w_H + r_tmp * r_H;
            }

            for (int h = 0; h < hidden_size; h++)
            {
                if (seq == 0)
                {
                    float Z = 1.f / (1.f + exp(-Z_data[h]));
                    float H = tanh(H_data[h]);
                    float out = (1 - Z) * H + Z * h_0[h];
                    output_data[seq * hidden_size * batch_size + t * hidden_size + h] = out;
                    output_h_data[t * hidden_size + h] = out;
                }
                else
                {
                    float Z = 1.f / (1.f + exp(-Z_data[h]));
                    float H = tanh(H_data[h]);
                    float out = (1 - Z) * H + Z * output_h_data[t * hidden_size + h];
                    output_data[seq * hidden_size * batch_size + t * hidden_size + h] = out;
                    output_h_data[t * hidden_size + h] = out;
                }
            }
        }
    }

    free(initial_h_data);
    free(output_h_data);
    free(h_0);
    free(Z_data);
    free(R_data);
    free(H_data);

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    // add by wangxinwei for gru op
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* w = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* r = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    struct tensor* b = NULL;
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (ir_node->input_num > 3)
        b = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[3]);

    struct gru_param* param = (struct gru_param*)(ir_node->op.param_mem);

    /* only support one way */
    if (w->dim_num == 4 && w->dims[0] == 2)
    {
        printf("GRU only support one way.\n");
        return -1;
    }

    int ret = -1;
    if (ir_node->input_num == 3)
    {
        ret = ref_gru_default_fp32(input_tensor, w, r, output_tensor, param);
    }
    else if (ir_node->input_num == 5)
    {
        ret = ref_gru_case1_fp32(input_tensor, w, r, b, output_tensor, param);
    }
    else
    {
        b = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[3]);
        ret = ref_gru_with_bias_fp32(input_tensor, w, r, b, output_tensor, param);
    }

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops gru_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_gru_ref_op()
{
    return register_builtin_node_ops(OP_GRU, &gru_node_ops);
}

int unregister_gru_ref_op()
{
    return unregister_builtin_node_ops(OP_GRU, &gru_node_ops);
}
