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
 * Author: zpluo@openailab.com
 */

#include "mvn_param.h"

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

typedef struct _ref_mvn_param
{
    int input_n;
    int input_h;
    int input_w;
    int input_c;
    int across_channels;
    int normalize_variance;
    float eps;
    int layout;
    float in_scale;
    int in_zero;
    float out_scale;
    int out_zero;
    float scale_scale;
    int scale_zero;
} ref_mvn_param, *p_ref_mvn_param;

int ref_mvn_fp32(float* in_data, float* out_data, p_ref_mvn_param param)
{
    int batch_num = param->input_n;
    int in_h = param->input_h;
    int in_w = param->input_w;
    int in_c = param->input_c;
    int in_size = in_h * in_w;
    int image_size = in_size * in_c;
    int offset = 0;
    int layout = param->layout;
    int across_channels = param->across_channels;
    int normalize_variance = param->normalize_variance;
    float eps = param->eps;

    float* sum = (float*)malloc(in_c * sizeof(float));

    if (NULL == sum)
        return -100;

    for (int n = 0; n < batch_num; n++)
    {
        for (int c = 0; c < in_c; c++)
        {
            float s = 0.f;
            for (int i = 0; i < in_size; i++)
            {
                if (TENGINE_LAYOUT_NCHW == layout)
                    offset = n * image_size + c * in_size + i;
                else
                    offset = n * image_size + i * in_c + c;
                s += in_data[offset];
            }
            sum[c] = s;
        }

        if (across_channels)
        {
            float mean = 0.f;
            for (int c = 0; c < in_c; c++)
            {
                mean += sum[c];
            }
            mean = mean / (in_size * in_c);

            for (int c = 0; c < in_c; c++)
            {
                for (int i = 0; i < in_size; i++)
                {
                    if (TENGINE_LAYOUT_NCHW == layout)
                        offset = n * image_size + c * in_size + i;
                    else
                        offset = n * image_size + i * in_c + c;
                    out_data[offset] = in_data[offset] - mean;
                }
            }
        }
        else
        {
            for (int c = 0; c < in_c; c++)
            {
                float mean = sum[c] / in_size;

                for (int i = 0; i < in_size; i++)
                {
                    if (TENGINE_LAYOUT_NCHW == layout)
                        offset = n * image_size + c * in_size + i;
                    else
                        offset = n * image_size + i * in_c + c;
                    out_data[offset] = in_data[offset] - mean;
                }
            }
        }

        if (normalize_variance)
        {
            float* sqsum = (float*)malloc(in_c * sizeof(float));
            if (NULL == sqsum)
                return -100;

            for (int c = 0; c < in_c; c++)
            {
                float s = 0.f;
                for (int i = 0; i < in_size; i++)
                {
                    if (TENGINE_LAYOUT_NCHW == layout)
                        offset = n * image_size + c * in_size + i;
                    else
                        offset = n * image_size + i * in_c + c;
                    s += in_data[offset] * in_data[offset];
                }
                sqsum[c] = s;
            }

            if (across_channels)
            {
                float sqmean = 0.f;
                for (int c = 0; c < in_c; c++)
                {
                    sqmean += sqsum[c];
                }
                sqmean = sqmean / (in_c * in_size);

                float norm_var = sqrt(sqmean) + eps;

                for (int c = 0; c < in_c; c++)
                {
                    for (int i = 0; i < in_size; i++)
                    {
                        if (TENGINE_LAYOUT_NCHW == layout)
                            offset = n * image_size + c * in_size + i;
                        else
                            offset = n * image_size + i * in_c + c;
                        out_data[offset] = out_data[offset] / norm_var;
                    }
                }
            }
            else
            {
                for (int c = 0; c < in_c; c++)
                {
                    float sqmean = sqsum[c] / in_size;
                    float norm_var = sqrt(sqmean) + eps;
                    for (int i = 0; i < in_size; i++)
                    {
                        if (TENGINE_LAYOUT_NCHW == layout)
                            offset = n * image_size + c * in_size + i;
                        else
                            offset = n * image_size + i * in_c + c;
                        out_data[offset] = out_data[offset] / norm_var;
                    }
                }
            }
            free(sqsum);
            sqsum = NULL;
        }
    }

    free(sum);
    sum = NULL;
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

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* graph = node->graph;

    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);

    ref_mvn_param op_param;

    op_param.input_n = input_tensor->dims[0];
    op_param.input_c = input_tensor->dims[1];
    op_param.input_h = input_tensor->dims[2];
    op_param.input_w = input_tensor->dims[3];

    struct mvn_param* param = (struct mvn_param*)node->op.param_mem;
    op_param.normalize_variance = param->normalize_variance;
    op_param.across_channels = param->across_channels;
    op_param.eps = param->eps;
    op_param.layout = graph->graph_layout;

    void* in_data = input_tensor->data;
    void* out_data = output_tensor->data;

    return ref_mvn_fp32((float*)in_data, (float*)out_data, &op_param);
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = false};

int register_mvn_ref_op()
{
    return register_builtin_node_ops(OP_MVN, &hcl_node_ops);
}

int unregister_mvn_ref_op()
{
    return unregister_builtin_node_ops(OP_MVN, &hcl_node_ops);
}
