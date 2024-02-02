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

#include "detection_output_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/vector.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>
#include <string.h>

typedef struct
{
    float x0;
    float y0;
    float x1;
    float y1;
    int class_idx;
    float score;
} Box_t;

#define T_MAX(a, b) ((a) > (b) ? (a) : (b))
#define T_MIN(a, b) ((a) < (b) ? (a) : (b))

static void quick_sort(Box_t* array, int left, int right)
{
    int i = left;
    int j = right;
    Box_t key;

    if (left >= right)
        return;

    memmove(&key, &array[left], sizeof(Box_t));
    while (left < right)
    {
        while (left < right && key.score >= array[right].score)
        {
            --right;
        }
        memmove(&array[left], &array[right], sizeof(Box_t));
        while (left < right && key.score <= array[left].score)
        {
            ++left;
        }
        memmove(&array[right], &array[left], sizeof(Box_t));
    }

    memmove(&array[left], &key, sizeof(Box_t));

    quick_sort(array, i, left - 1);
    quick_sort(array, left + 1, j);
}

static void get_boxes(Box_t* boxes, int num_prior, float* loc_ptr, float* prior_ptr)
{
    for (int i = 0; i < num_prior; i++)
    {
        float* loc = loc_ptr + i * 4;
        float* pbox = prior_ptr + i * 4;
        float* pvar = pbox + num_prior * 4;
        // center size
        // pbox [xmin,ymin,xmax,ymax]
        float pbox_w = pbox[2] - pbox[0];
        float pbox_h = pbox[3] - pbox[1];
        float pbox_cx = (pbox[0] + pbox[2]) * 0.5f;
        float pbox_cy = (pbox[1] + pbox[3]) * 0.5f;

        // loc []
        float bbox_cx = pvar[0] * loc[0] * pbox_w + pbox_cx;
        float bbox_cy = pvar[1] * loc[1] * pbox_h + pbox_cy;
        float bbox_w = pbox_w * expf(pvar[2] * loc[2]);
        float bbox_h = pbox_h * expf(pvar[3] * loc[3]);
        // bbox [xmin,ymin,xmax,ymax]
        boxes[i].x0 = bbox_cx - bbox_w * 0.5f;
        boxes[i].y0 = bbox_cy - bbox_h * 0.5f;
        boxes[i].x1 = bbox_cx + bbox_w * 0.5f;
        boxes[i].y1 = bbox_cy + bbox_h * 0.5f;
    }
}

static inline float intersection_area(const Box_t* a, const Box_t* b)
{
    if (a->x0 > b->x1 || a->x1 < b->x0 || a->y0 > b->y1 || a->y1 < b->y0)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = T_MIN(a->x1, b->x1) - T_MAX(a->x0, b->x0);
    float inter_height = T_MIN(a->y1, b->y1) - T_MAX(a->y0, b->y0);

    return inter_width * inter_height;
}

void nms_sorted_bboxes(const Box_t* bboxes, int bboxes_num, int* picked, int* picked_num, float nms_threshold)
{
    float* areas = (float*)sys_malloc(sizeof(float) * bboxes_num);

    for (int i = 0; i < bboxes_num; i++)
    {
        float width = bboxes[i].x1 - bboxes[i].x0;
        float height = bboxes[i].y1 - bboxes[i].y0;

        areas[i] = width * height;
    }

    for (int i = 0; i < bboxes_num; i++)
    {
        int keep = 1;
        for (int j = 0; j < *picked_num; j++)
        {
            // intersection over union
            float inter_area = intersection_area(&bboxes[i], &bboxes[picked[j]]);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            //             float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
        {
            picked[*picked_num] = i;
            *picked_num += 1;
        }
    }

    sys_free(areas);
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
    struct tensor* loc_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* conf_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* priorbox_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    detection_output_param_t* param = (detection_output_param_t*)(ir_node->op.param_mem);

    float* location = NULL;
    float* confidence = NULL;
    float* priorbox = NULL;

    /* use original fp32 data or dequant uint8 to fp32 */
    if (loc_tensor->data_type == TENGINE_DT_FP32)
        location = (float*)loc_tensor->data;
    else if (loc_tensor->data_type == TENGINE_DT_UINT8)
    {
        uint8_t* location_u8 = (uint8_t*)loc_tensor->data;
        uint32_t elem_num = loc_tensor->elem_num;
        uint32_t zero_point = loc_tensor->zero_point;
        float scale = loc_tensor->scale;
        location = (float*)sys_malloc(elem_num * sizeof(float));
        for (int i = 0; i < elem_num; i++)
        {
            location[i] = ((float)location_u8[i] - (float)zero_point) * scale;
        }
    }
    else if (loc_tensor->data_type == TENGINE_DT_INT8)
    {
        int8_t* location_i8 = (int8_t*)loc_tensor->data;
        uint32_t elem_num = loc_tensor->elem_num;
        float scale = loc_tensor->scale;
        location = (float*)sys_malloc(elem_num * sizeof(float));
        for (int i = 0; i < elem_num; i++)
        {
            location[i] = (float)location_i8[i] * scale;
        }
    }

    if (conf_tensor->data_type == TENGINE_DT_FP32)
        confidence = (float*)conf_tensor->data;
    else if (conf_tensor->data_type == TENGINE_DT_UINT8)
    {
        uint8_t* confidence_u8 = (uint8_t*)conf_tensor->data;
        uint32_t elem_num = conf_tensor->elem_num;
        uint32_t zero_point = conf_tensor->zero_point;
        float scale = conf_tensor->scale;
        confidence = (float*)sys_malloc(elem_num * sizeof(float));
        for (int i = 0; i < elem_num; i++)
        {
            confidence[i] = ((float)confidence_u8[i] - (float)zero_point) * scale;
        }
    }
    else if (conf_tensor->data_type == TENGINE_DT_INT8)
    {
        int8_t* confidence_i8 = (int8_t*)conf_tensor->data;
        uint32_t elem_num = conf_tensor->elem_num;
        float scale = conf_tensor->scale;
        confidence = (float*)sys_malloc(elem_num * sizeof(float));
        for (int i = 0; i < elem_num; i++)
        {
            confidence[i] = (float)confidence_i8[i] * scale;
        }
    }

    if (priorbox_tensor->data_type == TENGINE_DT_FP32)
        priorbox = (float*)priorbox_tensor->data;
    else if (priorbox_tensor->data_type == TENGINE_DT_UINT8)
    {
        uint8_t* priorbox_u8 = (uint8_t*)priorbox_tensor->data;
        uint32_t elem_num = priorbox_tensor->elem_num;
        uint32_t zero_point = priorbox_tensor->zero_point;
        float scale = priorbox_tensor->scale;
        priorbox = (float*)sys_malloc(elem_num * sizeof(float));
        for (int i = 0; i < elem_num; i++)
        {
            priorbox[i] = ((float)priorbox_u8[i] - (float)zero_point) * scale;
        }
    }
    else if (priorbox_tensor->data_type == TENGINE_DT_INT8)
    {
        int8_t* priorbox_i8 = (int8_t*)priorbox_tensor->data;
        uint32_t elem_num = priorbox_tensor->elem_num;
        float scale = priorbox_tensor->scale;
        priorbox = (float*)sys_malloc(elem_num * sizeof(float));
        for (int i = 0; i < elem_num; i++)
        {
            priorbox[i] = (float)priorbox_i8[i] * scale;
        }
    }

    const int num_priorx4 = priorbox_tensor->dims[2];
    const int num_prior = num_priorx4 / 4;
    const int num_classes = param->num_classes;

    int b = 0;
    float* loc_ptr = location + b * num_priorx4;
    float* conf_ptr = confidence + b * num_prior * num_classes;
    float* prior_ptr = priorbox + b * num_priorx4 * 2;

    Box_t* boxes = (Box_t*)sys_malloc(sizeof(Box_t) * num_prior);
    get_boxes(boxes, num_prior, loc_ptr, prior_ptr);
    struct vector* output_bbox_v = create_vector(sizeof(Box_t), NULL);

    for (int i = 1; i < num_classes; i++)
    {
        Box_t* class_box = (Box_t*)sys_malloc(sizeof(Box_t) * num_prior);
        int class_box_num = 0;
        for (int j = 0; j < num_prior; j++)
        {
            float score = conf_ptr[j * num_classes + i];
            if (score > param->confidence_threshold)
            {
                boxes[j].score = score;
                boxes[j].class_idx = i;
                memcpy(&class_box[class_box_num++], &boxes[j], sizeof(Box_t));
            }
        }

        quick_sort(class_box, 0, class_box_num - 1);

        if (class_box_num > param->nms_top_k)
            class_box_num = param->nms_top_k;

        int* picked = (int*)sys_malloc(sizeof(int) * class_box_num); // = NULL;
        int picked_num = 0;
        nms_sorted_bboxes(class_box, class_box_num, picked, &picked_num, param->nms_threshold);

        for (int j = 0; j < picked_num; j++)
        {
            int z = picked[j];
            push_vector_data(output_bbox_v, &class_box[z]);
        }

        sys_free(picked);
        sys_free(class_box);
    }

    sys_free(boxes);

    int total_num = get_vector_num(output_bbox_v);
    Box_t* bbox_rects = (Box_t*)sys_malloc(total_num * sizeof(Box_t));

    for (int i = 0; i < total_num; i++)
        memcpy(&bbox_rects[i], get_vector_data(output_bbox_v, i), sizeof(Box_t));

    quick_sort(bbox_rects, 0, total_num - 1);

    if (total_num > param->keep_top_k)
        total_num = param->keep_top_k;

    int num_detected = total_num;
    int dims[4] = {1, num_detected, 6, 1};
    set_ir_tensor_shape(output_tensor, dims, 4);

    // output
    float* output_fp32 = NULL;
    if (output_tensor->data_type == TENGINE_DT_FP32)
        output_fp32 = (float*)output_tensor->data;
    else
    {
        output_fp32 = (float*)sys_malloc(output_tensor->elem_num * sizeof(float));
    }

    for (int i = 0; i < num_detected; i++)
    {
        float* outptr = output_fp32 + i * 6;
        outptr[0] = bbox_rects[i].class_idx;
        outptr[1] = bbox_rects[i].score;
        outptr[2] = bbox_rects[i].x0;
        outptr[3] = bbox_rects[i].y0;
        outptr[4] = bbox_rects[i].x1;
        outptr[5] = bbox_rects[i].y1;
    }

    sys_free(bbox_rects);
    release_vector(output_bbox_v);

    /* quant uint8 */
    if (output_tensor->data_type == TENGINE_DT_UINT8)
    {
        uint8_t* output_u8 = (uint8_t*)output_tensor->data;
        uint32_t elem_num = output_tensor->elem_num;
        float scale = output_tensor->scale;
        uint32_t zero_point = output_tensor->zero_point;
        for (int i = 0; i < elem_num; i++)
        {
            int udata = (int)(output_fp32[i] / scale + zero_point);
            if (udata > 255)
                udata = 255;
            else if (udata < 0)
                udata = 0;

            output_u8[i] = udata;
        }

        sys_free(location);
        sys_free(confidence);
        sys_free(priorbox);
        sys_free(output_fp32);
    }
    /* quant int8 */
    else if (output_tensor->data_type == TENGINE_DT_INT8)
    {
        int8_t* output_i8 = (int8_t*)output_tensor->data;
        int32_t elem_num = output_tensor->elem_num;
        float scale = output_tensor->scale;
        for (int i = 0; i < elem_num; i++)
        {
            int data_i32 = round(output_fp32[i] / scale);
            if (data_i32 > 127)
                data_i32 = 127;
            else if (data_i32 < -127)
                data_i32 = -127;
            output_i8[i] = (int8_t)data_i32;
        }

        sys_free(location);
        sys_free(confidence);
        sys_free(priorbox);
        sys_free(output_fp32);
    }

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops detection_output_node_ops = {.prerun = NULL,
                                                    .run = run,
                                                    .reshape = NULL,
                                                    .postrun = NULL,
                                                    .init_node = init_node,
                                                    .release_node = release_node,
                                                    .score = score,
                                                    .is_ref_op = true};

int register_detection_output_ref_op()
{
    return register_builtin_node_ops(OP_DETECTION_OUTPUT, &detection_output_node_ops);
}

int unregister_detection_output_ref_op()
{
    return unregister_builtin_node_ops(OP_DETECTION_OUTPUT, &detection_output_node_ops);
}
