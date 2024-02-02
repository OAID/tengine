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

#include "detection_postprocess_param.h"

#include "convolution_param.h"

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

struct Dpp_Box
{
    float x0; // xmin
    float y0; // ymin
    float x1; // xmax
    float y1; // ymax
    int box_idx;
    int class_idx;
    float score;
};

struct dpp_param
{
    int max_detections;
    int max_classes_per_detection;
    float nms_score_threshold;
    float nms_iou_threshold;
    int num_classes;
    int num_boxes;
    float scales[4];
    float quant_scale[3];
    int zero[3];
};

#define DPP_MIN(a, b) (a < b ? a : b)
#define DPP_MAX(a, b) (a > b ? a : b)
static float intersection_area(const struct Dpp_Box a, const struct Dpp_Box b)
{
    if (a.x0 > b.x1 || a.x1 < b.x0 || a.y0 > b.y1 || a.y1 < b.y0)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = DPP_MIN(a.x1, b.x1) - DPP_MAX(a.x0, b.x0);
    float inter_height = DPP_MIN(a.y1, b.y1) - DPP_MAX(a.y0, b.y0);

    return inter_width * inter_height;
}

static void nms_sorted_bboxes(const struct Dpp_Box* boxes, int boxes_size, int* picked, int* picked_size,
                              float nms_threshold)
{
    float* areas = (float*)sys_malloc(sizeof(float) * boxes_size);
    int n_picked = 0;
    for (int i = 0; i < boxes_size; i++)
    {
        float width = boxes[i].x1 - boxes[i].x0;
        float height = boxes[i].y1 - boxes[i].y0;

        areas[i] = width * height;
    }

    for (int i = 0; i < boxes_size; i++)
    {
        int keep = 1;
        for (int j = 0; j < n_picked; j++)
        {
            // intersection over union
            float inter_area = intersection_area(boxes[i], boxes[picked[j]]);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
        {
            picked[n_picked] = i;
            n_picked++;
        }
    }
    *picked_size = n_picked;

    sys_free(areas);
}

static void sort_boxes_by_score(struct Dpp_Box* boxes, int size)
{
    int i, j;
    for (i = 0; i < size - 1; i++)
    {
        int max_idx = i;
        for (j = i + 1; j < size; j++)
        {
            if (boxes[j].score < 0.6)
                continue;
            if (boxes[max_idx].score < boxes[j].score)
                max_idx = j;
        }
        if (i != max_idx)
        {
            struct Dpp_Box tmp;
            memcpy(&tmp, boxes + i, sizeof(struct Dpp_Box));
            memcpy(boxes + i, boxes + max_idx, sizeof(struct Dpp_Box));
            memcpy(boxes + max_idx, &tmp, sizeof(struct Dpp_Box));
        }
        else
        {
            if (boxes[max_idx].score < 0.6)
                return;
        }
    }
}

static int decode_single_box(struct Dpp_Box* box, const float* box_ptr, const float* anchor_ptr,
                             const float* scales)
{
    int i = box->box_idx;

    const float* box_coord = box_ptr + i * 4;
    const float* anchor = anchor_ptr + i * 4;

    // [0]: y  [1]: x  [2]: h  [3]: w
    float ycenter = box_coord[0] / scales[0] * anchor[2] + anchor[0];
    float xcenter = box_coord[1] / scales[1] * anchor[3] + anchor[1];
    float half_h = 0.5f * (exp(box_coord[2] / scales[2])) * anchor[2];
    float half_w = 0.5f * (exp(box_coord[3] / scales[3])) * anchor[3];

    box->y0 = ycenter - half_h;
    box->x0 = xcenter - half_w;
    box->y1 = ycenter + half_h;
    box->x1 = xcenter + half_w;
    if (box->y0 < 0 || box->x0 < 0)
        return -1;
    return 0;
}

void get_all_boxes_rect(struct Dpp_Box* all_class_bbox_rects, const float* box, const float* scores,
                        const float* anchor, int num_boxes, int num_classes, float* scales)
{
    struct Dpp_Box selected_box;
    for (int j = 0; j < num_boxes; j++)
    {
        for (int i = 1; i < num_classes; i++)
        {
            float score = scores[j * num_classes + i];

            if (score < 0.6)
                continue;

            selected_box.score = score;
            selected_box.class_idx = i;
            selected_box.box_idx = j;

            if (decode_single_box(&selected_box, box, anchor, scales) < 0)
                continue;

            // struct Box* cls_vector = all_class_bbox_rects[i];
            memcpy(all_class_bbox_rects + i * num_boxes + j, &selected_box, sizeof(struct Dpp_Box));
        }
    }
}

int ref_dpp_fp32(const float* input_f, const float* score_f, const float* anchor_f,
                 float* detect_num, float* detect_class, float* detect_score, float* detect_boxes, struct dpp_param* param)
{
    const int num_classes = param->num_classes + 1;
    const int num_boxes = param->num_boxes;
    const int max_detections = param->max_detections;

    struct Dpp_Box* all_boxes = (struct Dpp_Box*)malloc((unsigned long)num_classes * num_boxes * sizeof(struct Dpp_Box));
    memset(all_boxes, 0, sizeof(struct Dpp_Box) * num_classes * num_boxes);

    get_all_boxes_rect(all_boxes, input_f, score_f, anchor_f, num_boxes, num_classes, param->scales);

    int max_picked_boxes = 2 * max_detections * num_classes;
    struct Dpp_Box* picked_boxes = (struct Dpp_Box*)malloc(max_picked_boxes * sizeof(struct Dpp_Box));
    memset(picked_boxes, 0, sizeof(struct Dpp_Box) * max_picked_boxes);
    int all_picked_size = 0;

    for (int i = 1; i < num_classes; i++)
    {
        struct Dpp_Box* class_box = all_boxes + i * num_boxes;

        // sort
        sort_boxes_by_score(class_box, num_boxes);
        int box_size = 0;
        for (int j = 0; j < num_boxes; j++)
        {
            if (class_box[j].score < 0.6)
                break;
            box_size++;
        }
        if (box_size == 0)
            continue;

        if (box_size > max_detections * 2)
            box_size = max_detections * 2;

        int* picked = (int*)sys_malloc(sizeof(int) * num_boxes);
        int picked_size = 0;

        picked[0] = 0;
        nms_sorted_bboxes(class_box, box_size, picked, &picked_size, param->nms_iou_threshold);

        // save the survivors
        for (int j = 0; j < picked_size; j++)
        {
            int z = picked[j];
            memcpy(picked_boxes + all_picked_size, class_box + z, sizeof(struct Dpp_Box));
            all_picked_size++;
        }

        sys_free(picked);
    }

    sort_boxes_by_score(picked_boxes, max_picked_boxes);
    if (all_picked_size > max_detections)
        all_picked_size = max_detections;

    // generate output tensors
    detect_num[0] = all_picked_size;

    for (int i = 0; i < all_picked_size; i++)
    {
        detect_class[i] = picked_boxes[i].class_idx;
        detect_score[i] = picked_boxes[i].score;

        detect_boxes[4 * i] = picked_boxes[i].x0;
        detect_boxes[4 * i + 1] = picked_boxes[i].y0;
        detect_boxes[4 * i + 2] = picked_boxes[i].x1;
        detect_boxes[4 * i + 3] = picked_boxes[i].y1;
    }

    free(all_boxes);
    free(picked_boxes);
    return 0;
}

int ref_dpp_uint8(const uint8_t* input, const uint8_t* score, const uint8_t* anchor,
                  float* detect_num, float* detect_class, float* detect_score, float* detect_boxes, struct dpp_param* param)
{
    const int num_classes = param->num_classes + 1;
    const int num_boxes = param->num_boxes;
    const int max_detections = param->max_detections;

    /* transform uint8_t to fp32 */
    int input_size = num_boxes * 4;
    int score_size = num_boxes * num_classes;
    float* input_f = (float*)malloc(input_size * sizeof(float));
    float* score_f = (float*)malloc(score_size * sizeof(float));
    float* anchor_f = (float*)malloc(input_size * sizeof(float));
    for (int i = 0; i < input_size; i++)
        input_f[i] = (input[i] - param->zero[0]) * param->quant_scale[0];
    for (int i = 0; i < score_size; i++)
        score_f[i] = score[i] * param->quant_scale[1];
    for (int i = 0; i < input_size; i++)
        anchor_f[i] = (anchor[i] - param->zero[2]) * param->quant_scale[2];

    struct Dpp_Box* all_boxes = (struct Dpp_Box*)malloc((unsigned long)num_classes * num_boxes * sizeof(struct Dpp_Box));
    memset(all_boxes, 0, sizeof(struct Dpp_Box) * num_classes * num_boxes);

    get_all_boxes_rect(all_boxes, input_f, score_f, anchor_f, num_boxes, num_classes, param->scales);

    int max_picked_boxes = 2 * max_detections * num_classes;
    struct Dpp_Box* picked_boxes = (struct Dpp_Box*)malloc(max_picked_boxes * sizeof(struct Dpp_Box));
    memset(picked_boxes, 0, sizeof(struct Dpp_Box) * max_picked_boxes);
    int all_picked_size = 0;

    for (int i = 1; i < num_classes; i++)
    {
        struct Dpp_Box* class_box = all_boxes + i * num_boxes;

        // sort
        sort_boxes_by_score(class_box, num_boxes);
        int box_size = 0;
        for (int j = 0; j < num_boxes; j++)
        {
            if (class_box[j].score < 0.6)
                break;
            box_size++;
        }
        if (box_size == 0)
            continue;

        if (box_size > max_detections * 2)
            box_size = max_detections * 2;

        int* picked = (int*)sys_malloc(sizeof(int) * num_boxes);
        int picked_size = 0;

        picked[0] = 0;
        nms_sorted_bboxes(class_box, box_size, picked, &picked_size, param->nms_iou_threshold);

        // save the survivors
        for (int j = 0; j < picked_size; j++)
        {
            int z = picked[j];
            memcpy(picked_boxes + all_picked_size, class_box + z, sizeof(struct Dpp_Box));
            all_picked_size++;
        }

        sys_free(picked);
    }

    sort_boxes_by_score(picked_boxes, max_picked_boxes);
    if (all_picked_size > max_detections)
        all_picked_size = max_detections;

    // generate output tensors
    detect_num[0] = all_picked_size;

    for (int i = 0; i < all_picked_size; i++)
    {
        detect_class[i] = picked_boxes[i].class_idx;
        detect_score[i] = picked_boxes[i].score;

        detect_boxes[4 * i] = picked_boxes[i].x0;
        detect_boxes[4 * i + 1] = picked_boxes[i].y0;
        detect_boxes[4 * i + 2] = picked_boxes[i].x1;
        detect_boxes[4 * i + 3] = picked_boxes[i].y1;
    }

    free(anchor_f);
    free(score_f);
    free(input_f);
    free(all_boxes);
    free(picked_boxes);

    return 0;
}

struct dpp_param param;
static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct detection_postprocess_param* param_ = (struct detection_postprocess_param*)ir_node->op.param_mem;

    param.max_classes_per_detection = param_->max_classes_per_detection;
    param.nms_iou_threshold = param_->nms_iou_threshold;
    param.nms_score_threshold = param_->nms_score_threshold;
    param.num_classes = param_->num_classes;
    param.max_detections = param_->max_detections;
    param.num_boxes = input_tensor->dims[2]; // h
    param.scales[0] = param_->scales[0];
    param.scales[1] = param_->scales[1];
    param.scales[2] = param_->scales[2];
    param.scales[3] = param_->scales[3];

    if (input_tensor->data_type != TENGINE_DT_FP32 && input_tensor->data_type != TENGINE_DT_FP16 && input_tensor->data_type != TENGINE_DT_UINT8)
    {
        TLOG_ERR("Not support!");
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;

    struct detection_postprocess_param* detection_postprocess_param = (struct detection_postprocess_param*)ir_node->op.param_mem;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    const void* input_data = input_tensor->data;
    struct tensor* score = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    void* score_data = score->data;
    struct tensor* anchor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    void* anchor_data = anchor->data;

    struct tensor* detect_boxes = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    float* detect_boxes_data = (float*)detect_boxes->data;
    struct tensor* detect_classes = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[1]);
    float* detect_classes_data = (float*)detect_classes->data;
    struct tensor* detect_scores = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[2]);
    float* detect_scores_data = (float*)detect_scores->data;
    struct tensor* detect_num = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[3]);
    float* detect_num_data = (float*)detect_num->data;

    if (input_tensor->data_type == TENGINE_DT_UINT8)
    {
        param.quant_scale[0] = input_tensor->scale;
        param.quant_scale[1] = score->scale;
        param.quant_scale[2] = anchor->scale;
        param.zero[0] = input_tensor->zero_point;
        param.zero[1] = score->zero_point;
        param.zero[2] = anchor->zero_point;
    }

    /* transpose nchw to nchw */
    if (input_tensor->dim_num == 3 && input_tensor->elem_size == 1)
    {
        int in_ch = input_tensor->dims[1];
        int in_w = input_tensor->dims[2];
        int in_size = input_tensor->elem_num;

        int score_ch = score->dims[1];
        int score_w = score->dims[2];
        int score_size = score->elem_num;

        uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
        uint8_t* score_uint8 = (uint8_t*)score->data;
        uint8_t* input_uint8_temp = (uint8_t*)malloc(in_size);
        uint8_t* score_uint8_temp = (uint8_t*)malloc(score_size);

        memcpy(input_uint8_temp, input_uint8, in_size);
        memcpy(score_uint8_temp, score_uint8, score_size);

        int index = 0;
        for (int w = 0; w < in_w; w++)
            for (int c = 0; c < in_ch; c++)
                input_uint8[index++] = input_uint8_temp[c * in_w + w];

        index = 0;
        for (int w = 0; w < score_w; w++)
            for (int c = 0; c < score_ch; c++)
                score_uint8[index++] = score_uint8_temp[c * score_w + w];

        free(input_uint8_temp);
        free(score_uint8_temp);
    }
    else
    {
        int in_ch = input_tensor->dims[1];
        int in_w = input_tensor->dims[2];
        int in_size = input_tensor->elem_num;

        int score_ch = score->dims[1];
        int score_w = score->dims[2];
        int score_size = score->elem_num;

        float* input_fp32 = (float*)input_tensor->data;
        float* score_fp32 = (float*)score->data;
        float* input_fp32_temp = (float*)malloc(in_size * sizeof(float));
        float* score_fp32_temp = (float*)malloc(score_size * sizeof(float));

        memcpy(input_fp32_temp, input_fp32, in_size);
        memcpy(score_fp32_temp, score_fp32, score_size);

        int index = 0;
        for (int w = 0; w < in_w; w++)
            for (int c = 0; c < in_ch; c++)
                input_fp32[index++] = input_fp32_temp[c * in_w + w];

        index = 0;
        for (int w = 0; w < score_w; w++)
            for (int c = 0; c < score_ch; c++)
                score_fp32[index++] = score_fp32_temp[c * score_w + w];

        free(input_fp32_temp);
        free(score_fp32_temp);
    }

    if (input_tensor->data_type == TENGINE_DT_FP32)
        ref_dpp_fp32((float*)input_data, (float*)score_data, (float*)anchor_data, detect_num_data,
                     detect_classes_data, detect_scores_data, detect_boxes_data, &param);
    else
        ref_dpp_uint8((uint8_t*)input_data, (uint8_t*)score_data, (uint8_t*)anchor_data, detect_num_data,
                      detect_classes_data, detect_scores_data, detect_boxes_data, &param);

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

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}
static struct node_ops detection_postprocess_node_ops = {.prerun = prerun,
                                                         .run = run,
                                                         .reshape = NULL,
                                                         .postrun = NULL,
                                                         .init_node = init_node,
                                                         .release_node = release_node,
                                                         .score = score,
                                                         .is_ref_op = true};

int register_detection_postprocess_ref_op()
{
    return register_builtin_node_ops(OP_DETECTION_POSTPROCESS, &detection_postprocess_node_ops);
}

int unregister_detection_postprocess_ref_op()
{
    unregister_builtin_node_ops(OP_DETECTION_POSTPROCESS, &detection_postprocess_node_ops);
    return 0;
}
