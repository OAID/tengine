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
 * Author: jxyang@openailab.com
 * 
 * original model: https://github.com/deepinsight/insightface/tree/master/RetinaFace#retinaface-pretrained-models
 */

/*
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/blob/master/examples/retinaface.cpp
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

#include <vector>
#include <string>

#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "common.h"

#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

#define MODEL_PATH "models/retinaface.tmfile"
#define IMAGE_PATH "images/selfie_960.jpg"

const float CONF_THRESH = 0.8f;
const float NMS_THRESH = 0.4f;

const char* input_name = "data";

const char* bbox_name[3] = {"face_rpn_bbox_pred_stride32", "face_rpn_bbox_pred_stride16", "face_rpn_bbox_pred_stride8"};
const char* score_name[3] = {"face_rpn_cls_prob_reshape_stride32", "face_rpn_cls_prob_reshape_stride16",
                             "face_rpn_cls_prob_reshape_stride8"};
const char* landmark_name[3] = {"face_rpn_landmark_pred_stride32", "face_rpn_landmark_pred_stride16",
                                "face_rpn_landmark_pred_stride8"};

const int stride[3] = {32, 16, 8};

const float g_scales[3][2] = {{32.f, 16.f}, {8.f, 4.f}, {2.f, 1.f}};

struct Size2i
{
    int width;
    int height;
};

struct Point2f
{
    float x;
    float y;
};

struct Box2f
{
    float x1;
    float y1;
    float x2;
    float y2;
};

struct Rect2f
{
    float x;
    float y;
    float w;
    float h;
};

struct Face2f
{
    float score;
    Rect2f rect;
    Point2f landmark[5];
};

void draw_target(const std::vector<Face2f>& all_pred_boxes, image img)
{
    const char* class_names[] = {"faces"};

    fprintf(stdout, "detected face num: %zu\n", all_pred_boxes.size());
    for (int b = 0; b < (int)all_pred_boxes.size(); b++)
    {
        Face2f box = all_pred_boxes[b];

        printf("BOX %.2f:( %g , %g ),( %g , %g )\n", box.score, box.rect.x, box.rect.y, box.rect.w, box.rect.h);

        draw_box(img, box.rect.x, box.rect.y, box.rect.x + box.rect.w, box.rect.y + box.rect.h, 2, 0, 255, 0);

        for (int l = 0; l < 5; l++)
        {
            draw_circle(img, box.landmark[l].x, box.landmark[l].y, 1, 0, 128, 128);
        }
    }
    save_image(img, "retinaface_out");
}

float iou(const Face2f& a, const Face2f& b)
{
    float area_a = a.rect.w * a.rect.h;
    float area_b = b.rect.w * b.rect.h;

    float xx1 = std::max(a.rect.x, b.rect.x);
    float yy1 = std::max(a.rect.y, b.rect.y);
    float xx2 = std::min(a.rect.x + a.rect.w, b.rect.x + b.rect.w);
    float yy2 = std::min(a.rect.y + a.rect.h, b.rect.y + b.rect.h);

    float w = std::max(float(0), xx2 - xx1 + 1);
    float h = std::max(float(0), yy2 - yy1 + 1);

    float inter = w * h;
    float ovr = inter / (area_a + area_b - inter);
    return ovr;
}

void nms_sorted_boxes(const std::vector<Face2f>& face_objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = face_objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = face_objects[i].rect.w * face_objects[i].rect.h;
    }

    for (int i = 0; i < n; i++)
    {
        const Face2f& a = face_objects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Face2f& b = face_objects[picked[j]];

            // intersection over union
            float inter_area = iou(a, b);
            if (inter_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void qsort_descent_inplace(std::vector<Face2f>& face_objects, const int& left, const int& right)
{
    int i = left;
    int j = right;

    float p = face_objects[(left + right) / 2].score;

    while (i <= j)
    {
        while (face_objects[i].score > p)
            i++;

        while (face_objects[j].score < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(face_objects[i], face_objects[j]);

            i++;
            j--;
        }
    }

    if (left < j)
        qsort_descent_inplace(face_objects, left, j);
    if (i < right)
        qsort_descent_inplace(face_objects, i, right);
}

void qsort_descent_inplace(std::vector<Face2f>& face_objects)
{
    if (face_objects.empty())
        return;

    qsort_descent_inplace(face_objects, 0, face_objects.size() - 1);
}

std::vector<Box2f> generate_anchors(int base_size, const std::vector<float>& ratios, const std::vector<float>& scales)
{
    size_t num_ratio = ratios.size();
    size_t num_scale = scales.size();

    std::vector<Box2f> anchors(num_ratio * num_scale);

    const float cx = (float)base_size * 0.5f;
    const float cy = (float)base_size * 0.5f;

    for (int i = 0; i < num_ratio; i++)
    {
        float ar = ratios[i];

        int r_w = (int)round((float)base_size / sqrt(ar));
        int r_h = (int)round((float)r_w * ar); // round(base_size * sqrt(ar));

        for (int j = 0; j < num_scale; j++)
        {
            float scale = scales[j];

            float rs_w = (float)r_w * scale;
            float rs_h = (float)r_h * scale;

            Box2f& anchor = anchors[i * num_scale + j];

            anchor.x1 = cx - rs_w * 0.5f;
            anchor.y1 = cy - rs_h * 0.5f;
            anchor.x2 = cx + rs_w * 0.5f;
            anchor.y2 = cy + rs_h * 0.5f;
        }
    }

    return anchors;
}

static void generate_proposals(std::vector<Box2f>& anchors, int feat_stride, const float* score_blob,
                               const int score_dims[], const float* bbox_blob, const int bbox_dims[],
                               const float* landmark_blob, const int landmark_dims[], const float& prob_threshold,
                               std::vector<Face2f>& faces)
{
    int w = bbox_dims[3];
    int h = bbox_dims[2];
    int offset = w * h;

    // generate face proposal from bbox deltas and shifted anchors
    const int num_anchors = anchors.size();

    for (int q = 0; q < num_anchors; q++)
    {
        const Box2f& anchor = anchors[q];

        const float* score = score_blob + (q + num_anchors) * offset;
        const float* bbox = bbox_blob + (q * 4) * offset;
        const float* landmark = landmark_blob + (q * 10) * offset;

        // shifted anchor
        float anchor_y = anchor.y1;

        float anchor_w = anchor.x2 - anchor.x1;
        float anchor_h = anchor.y2 - anchor.y1;

        for (int i = 0; i < h; i++)
        {
            float anchor_x = anchor.x1;

            for (int j = 0; j < w; j++)
            {
                int index = i * w + j;

                float prob = score[index];

                if (prob >= prob_threshold)
                {
                    // apply center size
                    float dx = bbox[index + offset * 0];
                    float dy = bbox[index + offset * 1];
                    float dw = bbox[index + offset * 2];
                    float dh = bbox[index + offset * 3];

                    float cx = anchor_x + anchor_w * 0.5f;
                    float cy = anchor_y + anchor_h * 0.5f;

                    float pb_cx = cx + anchor_w * dx;
                    float pb_cy = cy + anchor_h * dy;

                    float pb_w = anchor_w * exp(dw);
                    float pb_h = anchor_h * exp(dh);

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Face2f obj{};
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.w = x1 - x0 + 1;
                    obj.rect.h = y1 - y0 + 1;

                    obj.landmark[0].x = cx + (anchor_w + 1) * landmark[index + offset * 0];
                    obj.landmark[0].y = cy + (anchor_h + 1) * landmark[index + offset * 1];
                    obj.landmark[1].x = cx + (anchor_w + 1) * landmark[index + offset * 2];
                    obj.landmark[1].y = cy + (anchor_h + 1) * landmark[index + offset * 3];
                    obj.landmark[2].x = cx + (anchor_w + 1) * landmark[index + offset * 4];
                    obj.landmark[2].y = cy + (anchor_h + 1) * landmark[index + offset * 5];
                    obj.landmark[3].x = cx + (anchor_w + 1) * landmark[index + offset * 6];
                    obj.landmark[3].y = cy + (anchor_h + 1) * landmark[index + offset * 7];
                    obj.landmark[4].x = cx + (anchor_w + 1) * landmark[index + offset * 8];
                    obj.landmark[4].y = cy + (anchor_h + 1) * landmark[index + offset * 9];

                    obj.score = prob;

                    faces.push_back(obj);
                }

                anchor_x += (float)feat_stride;
            }

            anchor_y += (float)feat_stride;
        }
    }
}

int get_input_data(const char* image_file, std::vector<float>& image_data, Size2i& size)
{
    image img = imread(image_file);

    size.width = img.w;
    size.height = img.h;

    int img_size = img.w * img.h * img.c;

    img = image_permute(img);

    image_data.resize(img_size);

    memcpy(image_data.data(), img.data, img_size * sizeof(float));

    free_image(img);

    return img_size;
}

void show_usage()
{
    printf("[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count] [-n device_name]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;

    const char* model_file = MODEL_PATH;
    const char* image_file = IMAGE_PATH;
    const char* device_name = "";

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:n:")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'i':
            image_file = optarg;
            break;
        case 'r':
            repeat_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
            break;
        case 'n':
            device_name = optarg;
            break;
        case 'h':
            show_usage();
            return 0;
        default:
            break;
        }
    }

    /* check files */
    if (model_file == nullptr)
    {
        printf("Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (image_file == nullptr)
    {
        printf("Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    int ret = init_tengine();
    if (0 != ret)
    {
        printf("Init tengine-lite failed.\n");
        return -1;
    }

    printf("tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    context_t vk_context = create_context("VK", 1);
    add_context_device(vk_context, "VK");
    graph_t graph = create_graph(vk_context, "tengine", model_file);
    set_graph_device(graph, "VK");
    if (graph == nullptr)
    {
        printf("Load model to graph failed.\n");
        return -1;
    }

    /* prepare process input data */
    int target_size = 1024;
    int max_size = 1980;

    std::vector<float> image_data;

    Size2i image_size;
    // Size2i tensor_size;

    float im_scale;

    int img_size = get_input_data(image_file, image_data, image_size);

    /* set the input shape to initial the graph, and pre-run graph to infer shape */
    int dims[] = {1, 3, image_size.height, image_size.width};

    tensor_t input_tensor = get_graph_tensor(graph, input_name);
    if (nullptr == input_tensor)
    {
        printf("Get input tensor failed\n");
        return -1;
    }

    if (0 != set_tensor_shape(input_tensor, dims, 4))
    {
        printf("Set input tensor shape failed\n");
        return -1;
    }

    /* set the data mem to input tensor */
    if (set_tensor_buffer(input_tensor, image_data.data(), img_size * sizeof(float)) < 0)
    {
        printf("Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (0 != prerun_graph_multithread(graph, opt))
    {
        printf("Pre-run graph failed\n");
        return -1;
    }

    /* run graph */
    float min_time = FLT_MAX, max_time = 0, total_time = 0.f;
    for (int i = 0; i < repeat_count; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            printf("Run graph failed\n");
            return -1;
        }
        double end = get_current_time();

        float cur = float(end - start);

        total_time += cur;
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    printf("img_h, img_w : %d, %d\n", image_size.height, image_size.width);
    printf("Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
           num_thread, total_time / (float)repeat_count, max_time, min_time);
    printf("--------------------------------------\n");

    /* process the detection result */
    std::vector<Face2f> face_proposals;

    for (int stride_index = 0; stride_index < 3; stride_index++)
    {
        // ==================================================================
        // ========== This part is to get tensor information ================
        // ==================================================================
        tensor_t score_blob_tensor = get_graph_tensor(graph, score_name[stride_index]);
        tensor_t bbox_blob_tensor = get_graph_tensor(graph, bbox_name[stride_index]);
        tensor_t landmark_blob_tensor = get_graph_tensor(graph, landmark_name[stride_index]);

        int score_blob_dims[MAX_SHAPE_DIM_NUM] = {0};
        int bbox_blob_dims[MAX_SHAPE_DIM_NUM] = {0};
        int landmark_blob_dims[MAX_SHAPE_DIM_NUM] = {0};

        get_tensor_shape(score_blob_tensor, score_blob_dims, MAX_SHAPE_DIM_NUM);
        get_tensor_shape(bbox_blob_tensor, bbox_blob_dims, MAX_SHAPE_DIM_NUM);
        get_tensor_shape(landmark_blob_tensor, landmark_blob_dims, MAX_SHAPE_DIM_NUM);

        float* score_blob = (float*)get_tensor_buffer(score_blob_tensor);
        float* bbox_blob = (float*)get_tensor_buffer(bbox_blob_tensor);
        float* landmark_blob = (float*)get_tensor_buffer(landmark_blob_tensor);

        const int base_size = 16;
        const int feat_stride = stride[stride_index];

        std::vector<float> current_ratios(1);
        current_ratios[0] = 1.f;

        std::vector<float> current_scales(2);
        current_scales[0] = g_scales[stride_index][0];
        current_scales[1] = g_scales[stride_index][1];

        const float threshold = CONF_THRESH;

        std::vector<Box2f> anchors = generate_anchors(base_size, current_ratios, current_scales);

        std::vector<Face2f> face_objects;
        generate_proposals(anchors, feat_stride, score_blob, score_blob_dims, bbox_blob, bbox_blob_dims, landmark_blob,
                           landmark_blob_dims, threshold, face_objects);

        face_proposals.insert(face_proposals.end(), face_objects.begin(), face_objects.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(face_proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_boxes(face_proposals, picked, NMS_THRESH);

    int face_count = picked.size();

    std::vector<Face2f> face_objects(face_count);
    for (int i = 0; i < face_count; i++)
    {
        face_objects[i] = face_proposals[picked[i]];

        // clip to image size
        float x0 = face_objects[i].rect.x;
        float y0 = face_objects[i].rect.y;
        float x1 = x0 + face_objects[i].rect.w;
        float y1 = y0 + face_objects[i].rect.h;

        x0 = std::max(std::min(x0, (float)image_size.width - 1), 0.f);
        y0 = std::max(std::min(y0, (float)image_size.height - 1), 0.f);
        x1 = std::max(std::min(x1, (float)image_size.width - 1), 0.f);
        y1 = std::max(std::min(y1, (float)image_size.height - 1), 0.f);

        face_objects[i].rect.x = x0;
        face_objects[i].rect.y = y0;
        face_objects[i].rect.w = x1 - x0;
        face_objects[i].rect.h = y1 - y0;
    }

    image img = imread(image_file);
    draw_target(face_objects, img);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return 0;
}
