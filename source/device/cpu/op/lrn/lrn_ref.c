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
 * Author: qtang@openailab.com
 */

#include "lrn_param.h"

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

static int ref_lrn_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct lrn_param* param,
                        int num_thread)
{
    int n = input_tensor->dims[0];
    int c = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];

    float alpha = param->alpha;
    float beta = param->beta;
    float bias = param->k;
    int local_size = param->local_size;

    int channel_size = h * w;
    int img_size = c * channel_size;

    float* in_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    float* square = (float*)(malloc(img_size * sizeof(float)));
    float* accum_square = (float*)(malloc(channel_size * sizeof(float)));

    for (int i = 0; i < n; i++)
    {
        const float* img_base = in_data + i * img_size;

        /* get square value */
        for (int j = 0; j < img_size; j++)
            square[j] = img_base[j] * img_base[j] + bias;

        if (param->norm_region == 0) /* LRN_ACROSS_CHANNELS */
        {
            float alpha_over_size = alpha / local_size;

            for (int j = 0; j < c; j++)
            {
                int c_start = j - local_size / 2;
                int c_end = j + local_size / 2;

                memset(accum_square, 0x0, channel_size * sizeof(float));

                for (int l = c_start; l <= c_end; l++)
                {
                    if (l < 0 || l >= c)
                        continue;

                    for (int m = 0; m < channel_size; m++)
                    {
                        accum_square[m] += square[l * channel_size + m];
                    }
                }

                /* get the output */
                for (int m = 0; m < channel_size; m++)
                {
                    int offset = i * img_size + j * channel_size + m;
                    out_data[offset] = in_data[offset] * pow(1.0f + alpha_over_size * accum_square[m], -beta);
                }
            }
        }
        else
        {
            free(square);
            free(accum_square);
            return -1;
        }
    }

    free(square);
    free(accum_square);
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
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct lrn_param* lrn_param = (struct lrn_param*)ir_node->op.param_mem;

    ref_lrn_fp32(input_tensor, output_tensor, lrn_param, exec_graph->num_thread);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_lrn_ref_op()
{
    return register_builtin_node_ops(OP_LRN, &hcl_node_ops);
}

int unregister_lrn_ref_op()
{
    return unregister_builtin_node_ops(OP_LRN, &hcl_node_ops);
}
