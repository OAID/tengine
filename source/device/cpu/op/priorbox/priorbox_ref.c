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
 * Author: jxyang@openailab.com
 */

#include "priorbox_param.h"

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
    struct tensor* featmap_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* data_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    priorbox_param_t* param = (priorbox_param_t*)(ir_node->op.param_mem);

    float* output_fp32 = NULL;
    if (output_tensor->data_type == TENGINE_DT_FP32)
        output_fp32 = (float*)output_tensor->data;
    else if (output_tensor->data_type == TENGINE_DT_UINT8 || output_tensor->data_type == TENGINE_DT_INT8)
        output_fp32 = (float*)sys_malloc(output_tensor->elem_num * sizeof(float));

    const int data_height = data_tensor->dims[2];
    const int data_width = data_tensor->dims[3];
    const int feat_height = featmap_tensor->dims[2];
    const int feat_width = featmap_tensor->dims[3];
    int image_w, image_h;
    if (param->image_h == 0 || param->image_w == 0)
    {
        image_w = data_width;
        image_h = data_height;
    }
    else
    {
        image_w = param->image_w;
        image_h = param->image_h;
    }
    float step_w, step_h;
    if (param->step_h == 0 || param->step_w == 0)
    {
        step_w = (float)(image_w) / feat_width;
        step_h = (float)(image_h) / feat_height;
    }
    else
    {
        step_w = param->step_w;
        step_h = param->step_h;
    }
    // out shape [feat_width,feat_height,num_priors * 4,2]
    int num_priors = param->num_priors;

    // default offset=0.5
    // box[xmin,ymin,xmax,ymax]
    float offset_ = param->offset;
    for (int h = 0; h < feat_height; ++h)
    {
        float* box = output_fp32 + h * num_priors * 4 * feat_width;
        for (int w = 0; w < feat_width; ++w)
        {
            float center_x = (w + offset_) * step_w;
            float center_y = (h + offset_) * step_h;
            float box_width, box_height;
            for (int s = 0; s < (int)param->min_size_num; ++s)
            {
                int min_size_ = param->min_size[s];
                // first prior: aspect_ratio = 1, size = min_size
                box_width = box_height = min_size_;
                box[0] = (center_x - box_width * 0.5f) / image_w;
                box[1] = (center_y - box_height * 0.5f) / image_h;
                box[2] = (center_x + box_width * 0.5f) / image_w;
                box[3] = (center_y + box_height * 0.5f) / image_h;
                box += 4;

                // default：len(max_size)=len(min_size)
                if (param->max_size_num > 0)
                {
                    int max_size_ = param->max_size[s];
                    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    box_width = box_height = sqrt(min_size_ * max_size_);
                    box[0] = (center_x - box_width * 0.5f) / image_w;
                    box[1] = (center_y - box_height * 0.5f) / image_h;
                    box[2] = (center_x + box_width * 0.5f) / image_w;
                    box[3] = (center_y + box_height * 0.5f) / image_h;
                    box += 4;
                }

                // rest of priors
                for (int r = 0; r < (int)param->aspect_ratio_size; ++r)
                {
                    float ar = param->aspect_ratio[r];

                    box_width = min_size_ * sqrt(ar);
                    box_height = min_size_ / sqrt(ar);
                    box[0] = (center_x - box_width * 0.5f) / image_w;
                    box[1] = (center_y - box_height * 0.5f) / image_h;
                    box[2] = (center_x + box_width * 0.5f) / image_w;
                    box[3] = (center_y + box_height * 0.5f) / image_h;
                    box += 4;
                    if (param->flip)
                    {
                        box[0] = (center_x - box_height * 0.5f) / image_h;
                        box[1] = (center_y - box_width * 0.5f) / image_w;
                        box[2] = (center_x + box_height * 0.5f) / image_h;
                        box[3] = (center_y + box_width * 0.5f) / image_w;
                        box += 4;
                    }
                }
            }
        }
    }
    // clip the prior's coordidate such that it is within [0, 1]
    int dim = param->out_dim;
    if (param->clip)
    {
        for (int d = 0; d < dim; ++d)
        {
            output_fp32[d] = T_MIN(T_MAX(output_fp32[d], 0.f), 1.f);
        }
    }
    // set the variance.
    float* output_ptr = output_fp32 + dim;
    int size = dim / 4;
    for (int i = 0; i < size; i++)
    {
        output_ptr[0] = param->variance[0];
        output_ptr[1] = param->variance[1];
        output_ptr[2] = param->variance[2];
        output_ptr[3] = param->variance[3];
        output_ptr += 4;
    }

    /* quant to uint8 */
    if (output_tensor->data_type == TENGINE_DT_UINT8)
    {
        uint8_t* output_org = (uint8_t*)output_tensor->data;

        for (int i = 0; i < output_tensor->elem_num; i++)
        {
            int udata = (int)(output_fp32[i] / output_tensor->scale + output_tensor->zero_point);
            if (udata > 255)
                udata = 255;
            else if (udata < 0)
                udata = 0;

            output_org[i] = udata;
        }

        sys_free(output_fp32);
    }
    /* quant to int8 */
    if (output_tensor->data_type == TENGINE_DT_INT8)
    {
        int8_t* output_org = (int8_t*)output_tensor->data;

        for (int i = 0; i < output_tensor->elem_num; i++)
        {
            int data_i32 = round(output_fp32[i] / output_tensor->scale);
            if (data_i32 > 127)
                data_i32 = 127;
            else if (data_i32 < -127)
                data_i32 = -127;
            output_org[i] = (int8_t)data_i32;
        }

        sys_free(output_fp32);
    }

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops priorbox_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_priorbox_ref_op()
{
    return register_builtin_node_ops(OP_PRIORBOX, &priorbox_node_ops);
}

int unregister_priorbox_ref_op()
{
    return unregister_builtin_node_ops(OP_PRIORBOX, &priorbox_node_ops);
}
