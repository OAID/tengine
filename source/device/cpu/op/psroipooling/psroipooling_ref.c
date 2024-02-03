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
 * Author: chh@openailab.com
 */

#include "psroipooling_param.h"

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

#define T_MAX(a, b) ((a) > (b) ? (a) : (b))
#define T_MIN(a, b) ((a) < (b) ? (a) : (b))

static int ref_psroipooling_fp32(struct tensor* featmap_tensor, struct tensor* roi_tensor,
                                 struct tensor* output_tensor, struct psroipooling_param* param, int num_thread)
{
    float* featmap = (float*)featmap_tensor->data;
    float* roi = (float*)roi_tensor->data;
    float* output = (float*)output_tensor->data;

    float spatial_scale = param->spatial_scale;
    int out_h = param->pooled_h;
    int out_w = param->pooled_w;
    int channel = featmap_tensor->dims[1];
    int in_h = featmap_tensor->dims[2];
    int in_w = featmap_tensor->dims[3];
    int num_rois = roi_tensor->dims[1];
    int output_dim = param->output_dim;
    int pool_hw = out_h * out_w;

    for (int n = 0; n < num_rois; ++n)
    {
        float* roi_ptr = roi + n * 4;
        float roi_x0 = round(roi_ptr[0]) * spatial_scale;
        float roi_y0 = round(roi_ptr[1]) * spatial_scale;
        float roi_x1 = round(roi_ptr[2] + 1.f) * spatial_scale;
        float roi_y1 = round(roi_ptr[3] + 1.f) * spatial_scale;

        int roi_w = T_MAX(roi_x1 - roi_x0, 0);
        int roi_h = T_MAX(roi_y1 - roi_y0, 0);

        float bin_w = (float)roi_w / (float)out_w;
        float bin_h = (float)roi_h / (float)out_h;

        for (int c = 0; c < output_dim; c++)
        {
            float* outptr = output + c * pool_hw;
            for (int h = 0; h < out_h; ++h)
            {
                for (int w = 0; w < out_w; ++w)
                {
                    float* inptr = featmap + (c * out_h + h) * out_w + w;

                    int hstart = floor(roi_y0 + (float)(h)*bin_h);
                    int wstart = floor(roi_x0 + (float)(w)*bin_w);
                    int hend = ceil(roi_y0 + (float)(h + 1) * bin_h);
                    int wend = ceil(roi_x0 + (float)(w + 1) * bin_w);

                    hstart = T_MIN(T_MAX(hstart, 0), in_h);
                    wstart = T_MIN(T_MAX(wstart, 0), in_w);
                    hend = T_MIN(T_MAX(hend, 0), in_h);
                    wend = T_MIN(T_MAX(wend, 0), in_w);

                    int is_empty = (hend <= hstart) || (wend <= wstart);
                    int area = (hend - hstart) * (wend - wstart);

                    float sum = 0.f;
                    for (int y = hstart; y < hend; y++)
                    {
                        for (int x = wstart; x < wend; x++)
                        {
                            int index = y * in_w + x;
                            sum += inptr[index];
                        }
                    }
                    outptr[w] = is_empty ? 0.f : (sum / (float)area);
                }
                outptr += out_w;
            }
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
    struct tensor* featmap_tensor;
    struct tensor* roi_tensor;
    struct tensor* output_tensor;

    featmap_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    roi_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct psroipooling_param* psroipooling_param = (struct psroipooling_param*)ir_node->op.param_mem;

    ref_psroipooling_fp32(featmap_tensor, roi_tensor, output_tensor, psroipooling_param, exec_graph->num_thread);

    return 0;
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
                                       .is_ref_op = true};

int register_psroipooling_ref_op()
{
    return register_builtin_node_ops(OP_PSROIPOOLING, &hcl_node_ops);
}

int unregister_psroipooling_ref_op()
{
    return unregister_builtin_node_ops(OP_PSROIPOOLING, &hcl_node_ops);
}
