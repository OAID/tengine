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

#include "rpn_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/vector.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>
#include <string.h>

struct anchor_box
{
    float x0; // xmin
    float y0; // ymin
    float x1; // xmax
    float y1; // ymax
};
struct RPN_Box
{
    float x0; // xmin
    float y0; // ymin
    float x1; // xmax
    float y1; // ymax
    float score;
};

struct rpn_param_ref
{
    int feat_height;
    int feat_width;
    int feat_chan;
    int score_chan;
    float src_scale;
    int src_width;
    int src_height;
    int num_anchors;
    int min_size;
    int feat_stride;
    int per_nms_topn;
    int post_nms_topn;
    float nms_thresh;
    // float scales[4];
    // float quant_scale[3];
    // int zero[3];
};

#define RPN_MIN(a, b) ((a) < (b) ? (a) : (b))
#define RPN_MAX(a, b) ((a) > (b) ? (a) : (b))

static inline void bbox_tranform_inv(float* m_box, float* local_anchors, struct rpn_param_ref* param)
{
    int feat_size = param->feat_height * param->feat_width;
    int c_4 = param->feat_chan / 4;
    for (int i = 0; i < c_4; ++i)
    {
        for (int j = 0; j < (2 * feat_size); ++j)
        {
            local_anchors[(i * 4 + 2) * feat_size + j] -= local_anchors[(i * 4 + 0) * feat_size + j] - 1;
            local_anchors[(i * 4 + 0) * feat_size + j] += local_anchors[(i * 4 + 2) * feat_size + j] * 0.5;

            m_box[(i * 4 + 0) * feat_size + j] *= local_anchors[(i * 4 + 2) * feat_size + j];
            m_box[(i * 4 + 0) * feat_size + j] += local_anchors[(i * 4 + 0) * feat_size + j];

            m_box[(i * 4 + 2) * feat_size + j] = exp(m_box[(i * 4 + 2) * feat_size + j]);
            m_box[(i * 4 + 2) * feat_size + j] *= local_anchors[(i * 4 + 2) * feat_size + j];
        }
    }
}

static inline void ref_filter_boxes(struct RPN_Box* boxes, const float* featmap, const float* score, int* num_boxes,
                                    struct rpn_param_ref* param)
{
    float local_minsize = param->min_size * param->src_scale;
    int c_4 = param->feat_chan / 4;
    int feat_size = param->feat_height * param->feat_width;

    int offset_w, offset_h, offset_x, offset_y, offset_s;

    int num = 0;
    for (int h = 0; h < param->feat_height; h++)
    {
        for (int w = 0; w < param->feat_width; w++)
        {
            offset_x = h * param->feat_width + w;
            offset_y = offset_x + feat_size;
            offset_w = offset_y + feat_size;
            offset_h = offset_w + feat_size;
            offset_s = feat_size * param->num_anchors + offset_x;
            for (int c = 0; c < c_4; c++)
            {
                float width = featmap[offset_w];
                float height = featmap[offset_h];

                if ((width >= local_minsize) & (height >= local_minsize))
                {
                    struct RPN_Box tmp;
                    tmp.x0 = featmap[offset_x] - 0.5 * width;
                    tmp.y0 = featmap[offset_y] - 0.5 * height;
                    tmp.x1 = featmap[offset_x] + 0.5 * width;
                    tmp.y1 = featmap[offset_y] + 0.5 * height;
                    tmp.x0 = RPN_MIN(RPN_MAX(tmp.x0, 0), param->src_width);
                    tmp.y0 = RPN_MIN(RPN_MAX(tmp.y0, 0), param->src_height);
                    tmp.x1 = RPN_MIN(RPN_MAX(tmp.x1, 0), param->src_width);
                    tmp.y1 = RPN_MIN(RPN_MAX(tmp.y1, 0), param->src_height);
                    tmp.score = score[offset_s];
                    memcpy(boxes + num, &tmp, sizeof(struct RPN_Box));
                    num++;
                }
                offset_x += 4 * feat_size;
                offset_y += 4 * feat_size;
                offset_w += 4 * feat_size;
                offset_h += 4 * feat_size;
                offset_s += feat_size;
            }
        }
    }
    *num_boxes = num;
}

void sort_rpn_boxes_by_score(struct RPN_Box* boxes, int size)
{
    int i, j;
    for (i = 0; i < size - 1; i++)
    {
        int max_idx = i;
        for (j = i + 1; j < size; j++)
        {
            if (boxes[max_idx].score < boxes[j].score)
                max_idx = j;
        }
        if (i != max_idx)
        {
            struct RPN_Box tmp;
            memcpy(&tmp, boxes + i, sizeof(struct RPN_Box));
            memcpy(boxes + i, boxes + max_idx, sizeof(struct RPN_Box));
            memcpy(boxes + max_idx, &tmp, sizeof(struct RPN_Box));
        }
    }
}

void nms_rpn_boxes(struct RPN_Box* input_boxes, int* size, float nms_thresh)
{
    int input_size = *size;
    int output_size = 0;

    struct RPN_Box* output_boxes = (struct RPN_Box*)sys_malloc(sizeof(struct RPN_Box) * input_size);
    float* areas = (float*)sys_malloc(sizeof(float) * input_size);
    int* picked = (int*)sys_malloc(sizeof(int) * input_size);

    for (int i = 0; i < input_size; ++i)
    {
        areas[i] = (input_boxes[i].x1 - input_boxes[i].x0 + 1) * (input_boxes[i].y1 - input_boxes[i].y0 + 1);
    }
    for (int i = 0; i < input_size; ++i)
    {
        int keep = 1;
        for (int j = 0; j < output_size; j++)
        {
            float xx1 = RPN_MAX(input_boxes[i].x0, output_boxes[j].x0);
            float yy1 = RPN_MAX(input_boxes[i].y0, output_boxes[j].y0);
            float xx2 = RPN_MIN(input_boxes[i].x1, output_boxes[j].x1);
            float yy2 = RPN_MIN(input_boxes[i].y1, output_boxes[j].y1);
            float w = RPN_MAX(0.f, xx2 - xx1 + 1);
            float h = RPN_MAX(0.f, yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[picked[j]] - inter);

            if (ovr >= nms_thresh)
            {
                keep = 0;
                break;
            }
        }
        if (keep)
        {
            memcpy(output_boxes + output_size, input_boxes + i, sizeof(struct RPN_Box));
            picked[output_size] = i;
            output_size++;
        }
    }
    memcpy(input_boxes, output_boxes, output_size * sizeof(struct RPN_Box));
    *size = output_size;
    sys_free(picked);
    sys_free(areas);
    sys_free(output_boxes);
}

void ref_proposal_local_anchor(int feat_height, int feat_width, int feat_stride, struct vector* anchors,
                               float* local_anchors)
{
    int feat_size = feat_height * feat_width;
    int num_anchors = (int)anchors->elem_num;
    for (int i = 0; i < num_anchors; ++i)
    {
        for (int j = 0; j < feat_height; j++)
            for (int k = 0; k < feat_width; k++)
            {
                Anchor_t anchor_val = *(Anchor_t*)(get_vector_data(anchors, i));
                local_anchors[(i * 4 + 0) * feat_size + j * feat_width + k] = anchor_val.x0 + k * feat_stride;
                local_anchors[(i * 4 + 1) * feat_size + j * feat_width + k] = anchor_val.y0 + j * feat_stride;
                local_anchors[(i * 4 + 2) * feat_size + j * feat_width + k] = anchor_val.x1 + k * feat_stride;
                local_anchors[(i * 4 + 3) * feat_size + j * feat_width + k] = anchor_val.y1 + j * feat_stride;
            }
    }
}

int ref_rpn_fp32(const float* score, float* featmap, float* anchors, float* output, struct rpn_param_ref* param)
{
    if (score == NULL || featmap == NULL || anchors == NULL || output == NULL)
        return -1;
    int featmap_size = param->feat_height * param->feat_width * param->feat_chan;
    int max_num_boxes = featmap_size / 4;

    struct RPN_Box* boxes = (struct RPN_Box*)sys_malloc(max_num_boxes * sizeof(struct RPN_Box));

    bbox_tranform_inv(featmap, anchors, param);

    int num_boxes = 0;
    ref_filter_boxes(boxes, featmap, score, &num_boxes, param);

    sort_rpn_boxes_by_score(boxes, num_boxes);

    if (param->per_nms_topn > 0)
    {
        num_boxes = RPN_MIN(param->per_nms_topn, num_boxes);
    }
    nms_rpn_boxes(boxes, &num_boxes, param->nms_thresh);

    if (param->post_nms_topn > 0)
    {
        num_boxes = RPN_MIN(param->post_nms_topn, num_boxes);
    }
    // inder shape [default batch=1]

    // std::cout<<"num_box "<<num_box<<"\n";
    for (int i = 0; i < num_boxes; i++)
    {
        float* outptr = output + i * 4;
        outptr[0] = boxes[i].x0;
        outptr[1] = boxes[i].y0;
        outptr[2] = boxes[i].x1;
        outptr[3] = boxes[i].y1;
    }

    sys_free(boxes);
    return num_boxes;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    exec_node->inplace_map[0] = 0;
    exec_node->inplace_map[1] = 1;
    exec_node->inplace_map_num = 1;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    exec_node->inplace_map_num = 0;

    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    rpn_param_t* _param = (struct rpn_param*)(ir_node->op.param_mem);
    struct graph* ir_graph = ir_node->graph;
    struct tensor* score_tensor;
    struct tensor* featmap_tensor;
    struct tensor* info_tensor;
    struct tensor* output_tensor;

    score_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    featmap_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    info_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    const void* score_org = score_tensor->data;
    void* featmap_org = featmap_tensor->data;
    const float* info_org = (float*)info_tensor->data;
    void* output_org = output_tensor->data;

    struct rpn_param_ref param;
    param.num_anchors = (int)_param->anchors_->elem_num;
    param.feat_chan = featmap_tensor->dims[1];
    param.feat_height = featmap_tensor->dims[2];
    param.feat_width = featmap_tensor->dims[3];
    int feat_size = featmap_tensor->dims[2] * featmap_tensor->dims[3];
    param.score_chan = score_tensor->dims[1];
    param.src_height = info_org[0];
    param.src_width = info_org[1];
    param.src_scale = info_org[2];
    param.nms_thresh = _param->nms_thresh;
    param.post_nms_topn = _param->post_nms_topn;
    param.per_nms_topn = _param->per_nms_topn;
    param.min_size = _param->min_size;
    param.feat_stride = _param->feat_stride;
    int size = param.num_anchors * 4 * feat_size;
    float* local_anchors = (float*)sys_malloc(size * sizeof(float));

    ref_proposal_local_anchor(featmap_tensor->dims[2], featmap_tensor->dims[3], _param->feat_stride, _param->anchors_,
                              local_anchors);

    int output_num = ref_rpn_fp32((float*)score_org, (float*)featmap_org, local_anchors, (float*)output_org, &param);
    int dims[4];
    dims[0] = featmap_tensor->dims[0];
    dims[1] = output_num;
    dims[2] = 4;
    dims[3] = 1;

    sys_free(local_anchors);

    int ret = set_ir_tensor_shape(output_tensor, dims, 4);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops rpn_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_rpn_ref_op()
{
    return register_builtin_node_ops(OP_RPN, &rpn_node_ops);
}

int unregister_rpn_ref_op()
{
    return unregister_builtin_node_ops(OP_RPN, &rpn_node_ops);
}
