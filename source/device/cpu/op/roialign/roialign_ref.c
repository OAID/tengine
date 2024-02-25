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

#include "roialign_param.h"

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

static inline float bilinear_interpolate(const float* ptr, int w, int h, float x, float y)
{
    int x0 = x;
    int x1 = x0 + 1;
    int y0 = y;
    int y1 = y0 + 1;

    float a0 = x1 - x;
    float a1 = x - x0;
    float b0 = y1 - y;
    float b1 = y - y0;

    if (x1 >= w)
    {
        x1 = w - 1;
        a0 = 1.f;
        a1 = 0.f;
    }
    if (y1 >= h)
    {
        y1 = h - 1;
        b0 = 1.f;
        b1 = 0.f;
    }

    float r0 = ptr[y0 * w + x0] * a0 + ptr[y0 * w + x1] * a1;
    float r1 = ptr[y1 * w + x0] * a0 + ptr[y1 * w + x1] * a1;
    float v = r0 * b0 + r1 * b1;

    return v;
}

static int ref_roialign_fp32(struct tensor* input_tensor, struct tensor* roi_tensor,
                             struct tensor* output_tensor, struct roialign_param* param, int num_thread)
{
    float* data_in = (float*)input_tensor->data;
    float* roi_ptr = (float*)roi_tensor->data;
    float* data_out = (float*)output_tensor->data;

    int size = input_tensor->dims[0] * input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];

    int w = param->pooled_width;
    int h = param->pooled_height;
    float spatial_scale = param->spatial_scale;

    // TLOG_ERR("spatial %f \n", spatial_scale);

    float roi_x1 = roi_ptr[0] * spatial_scale;
    float roi_y1 = roi_ptr[1] * spatial_scale;
    float roi_x2 = roi_ptr[2] * spatial_scale;
    float roi_y2 = roi_ptr[3] * spatial_scale;

    float roi_w = T_MAX(roi_x2 - roi_x1, 1);
    float roi_h = T_MAX(roi_y2 - roi_y1, 1);

    float bin_size_w = roi_w / (float)w;
    float bin_size_h = roi_h / (float)h;

    int channel = input_tensor->dims[1];
    int in_height = input_tensor->dims[2];
    int in_width = input_tensor->dims[3];
    int out_height = output_tensor->dims[2];
    int out_width = output_tensor->dims[3];

    int inDataHW = in_height * in_width;
    // TLOG_ERR("in height, width : %d %d \n",param->in_height, param->in_width);
    int outDataHW = out_height * out_width;
    // TLOG_ERR("out height, width : %d %d \n",param->out_height, param->out_width);
    // TLOG_ERR("%f %f \n", bin_size_h, bin_size_w);

    for (int q = 0; q < channel; q++)
    {
        float* ptr = data_in + q * inDataHW;
        float* outptr = data_out + q * outDataHW;
        for (int ph = 0; ph < h; ph++)
        {
            for (int pw = 0; pw < w; pw++)
            {
                float hstart = roi_y1 + ph * bin_size_h;
                float wstart = roi_x1 + pw * bin_size_w;
                float hend = roi_y1 + (ph + 1) * bin_size_h;
                float wend = roi_x1 + (pw + 1) * bin_size_w;

                hstart = T_MIN(T_MAX(hstart, 0.f), (float)in_height);
                wstart = T_MIN(T_MAX(wstart, 0.f), (float)in_width);
                hend = T_MIN(T_MAX(hend, 0.f), (float)in_height);
                wend = T_MIN(T_MAX(wend, 0.f), (float)in_width);

                int bin_grid_h = ceil(hend - hstart);
                int bin_grid_w = ceil(wend - wstart);

                int is_empty = (hend <= hstart) || (wend <= wstart);
                int area = bin_grid_h * bin_grid_w;

                float sum = 0.f;
                for (int by = 0; by < bin_grid_h; by++)
                {
                    float y = hstart + (by + 0.5f) * bin_size_h / (float)bin_grid_h;

                    for (int bx = 0; bx < bin_grid_w; bx++)
                    {
                        float x = wstart + (bx + 0.5f) * bin_size_w / (float)bin_grid_w;

                        // bilinear interpolate at (x,y)
                        float v = bilinear_interpolate(ptr, in_width, in_height, x, y);
                        sum += v;
                    }
                }
                outptr[pw] = is_empty ? 0.f : (sum / (float)area);
            }
            outptr += w;
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
    struct roialign_param* roialign_param = (struct roialign_param*)ir_node->op.param_mem;

    ref_roialign_fp32(input_tensor, roi_tensor, output_tensor, roialign_param, exec_graph->num_thread);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_roialign_ref_op()
{
    return register_builtin_node_ops(OP_ROIALIGN, &hcl_node_ops);
}

int unregister_roialign_ref_op()
{
    return unregister_builtin_node_ops(OP_ROIALIGN, &hcl_node_ops);
}
