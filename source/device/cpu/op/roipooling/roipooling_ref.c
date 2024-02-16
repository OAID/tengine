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

#include "roipooling_param.h"

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

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static int ref_roipooling_fp32(struct tensor* input_tensor, struct tensor* roi_tensor,
                               struct tensor* output_tensor, struct roipooling_param* param, int num_thread)
{
    int in_c = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];

    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];

    int channel = in_c;
    int feat_size = in_h * in_w;
    int pool_hw = param->pooled_h * param->pooled_w;

    int num_rois = output_tensor->dims[0];

    float* featmap = (float*)input_tensor->data;
    float* roi = (float*)roi_tensor->data;
    float* output = (float*)output_tensor->data;

    for (int n = 0; n < num_rois; ++n)
    {
        float* roi_ptr = roi + n * 4;

        int roi_x0 = round((double)roi_ptr[0] * param->spatial_scale);
        int roi_y0 = round((double)roi_ptr[1] * param->spatial_scale);
        int roi_x1 = round((double)roi_ptr[2] * param->spatial_scale);
        int roi_y1 = round((double)roi_ptr[3] * param->spatial_scale);

        int roi_w = MAX(roi_x1 - roi_x0 + 1, 1);
        int roi_h = MAX(roi_y1 - roi_y0 + 1, 1);

        float bin_w = (float)roi_w / (float)out_w;
        float bin_h = (float)roi_h / (float)out_h;

        for (int c = 0; c < channel; ++c)
        {
            const float* feat_ptr = featmap + c * feat_size;

            for (int h = 0; h < out_h; ++h)
            {
                for (int w = 0; w < out_w; ++w)
                {
                    int h0 = roi_y0 + (int)floor((double)(h)*bin_h);
                    int h1 = roi_y0 + (int)ceil((double)(h + 1) * bin_h);
                    int w0 = roi_x0 + (int)floor((double)(w)*bin_w);
                    int w1 = roi_x0 + (int)ceil((double)(w + 1) * bin_w);

                    h0 = MIN(MAX(h0, 0), in_h);
                    h1 = MIN(MAX(h1, 0), in_h);
                    w0 = MIN(MAX(w0, 0), in_w);
                    w1 = MIN(MAX(w1, 0), in_w);

                    // bool is_empty = (h1 <= h0) || (w1 <= w0);
                    float max_value = (h1 <= h0) || (w1 <= w0) ? 0.f : feat_ptr[h0 * in_w + w0];
                    for (int y = h0; y < h1; y++)
                    {
                        for (int x = w0; x < w1; x++)
                        {
                            int idx = y * in_w + x;
                            max_value = MAX(max_value, feat_ptr[idx]);
                        }
                    }
                    output[h * out_w + w] = max_value;
                }
            }
            output += pool_hw;
        }
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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* roi_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    roi_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct roipooling_param* roipooling_param = (struct roipooling_param*)ir_node->op.param_mem;

    // set output dims
    int dims[4];
    dims[0] = roi_tensor->dims[1];
    dims[1] = input_tensor->dims[1];
    dims[2] = roipooling_param->pooled_h;
    dims[3] = roipooling_param->pooled_w;

    set_ir_tensor_shape(output_tensor, dims, 4);

    ref_roipooling_fp32(input_tensor, roi_tensor, output_tensor, roipooling_param, exec_graph->num_thread);

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);
    struct roipooling_param* roipooling_param = (struct roipooling_param*)node->op.param_mem;

    int dims[4];

    dims[0] = input->dims[0];
    dims[1] = input->dims[1];
    dims[2] = roipooling_param->pooled_h;
    dims[3] = roipooling_param->pooled_w;

    int ret = set_ir_tensor_shape(output, dims, 4);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_roipooling_ref_op()
{
    return register_builtin_node_ops(OP_ROIPOOLING, &hcl_node_ops);
}

int unregister_roipooling_ref_op()
{
    return unregister_builtin_node_ops(OP_ROIPOOLING, &hcl_node_ops);
}
