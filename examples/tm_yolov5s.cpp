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
 * Author: xwwang@openailab.com
 */

#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"
#include "tengine_c_api.h"
#include "tengine_operations.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void get_input_data_focas(const char* image_file, float* input_data, int img_h, int img_w, const float* mean, const float* scale)
{
    float means[3] = {means[0], means[1], means[2]};
    image img = imread(image_file);

    image resImg = make_image(img_w, img_h, img.c);
    tengine_resize_f32(img.data, resImg.data, img_w, img_h, img.c, img.h, img.w);

    image focas_img = make_image(int(img_w / 2), int(img_h / 2), 12);
    int focas_w = img_w / 2;
    int focas_h = img_h / 2;
    int new_index = 0;
    int old_index = 0;

    for (int c = 0; c <= 2; c++)
    {
        for (int h = 0; h <= img_h - 1; h++)
        {
            for (int w = 0; w <= img_w - 1; w++)
            {
                resImg.data[c * img_h * img_w + h * img_w + w] -= mean[c];
                resImg.data[c * img_h * img_w + h * img_w + w] /= 255.0;
            }
        }
    }
    //img = rgb2bgr_premute(img);
    for (int i = 0; i < 2; i++)
    {
        for (int g = 0; g < 2; g++)
        {
            for (int c = 0; c < resImg.c; c++)
            {
                for (int w = 0; w < focas_w; w++)
                {
                    for (int h = 0; h < focas_h; h++)
                    {
                        old_index = i + g * resImg.h + c * resImg.h * resImg.w + w * 2 * resImg.h + h * 2;
                        focas_img.data[new_index] = resImg.data[old_index];
                        new_index++;
                    }
                }
            }
        }
    }

    memcpy(input_data, focas_img.data, sizeof(float) * img.c * img_w * img_h);
    free_image(img);
    free_image(resImg);
    free_image(focas_img);
}

static void generate_proposals(int stride,  const float* feat, float prob_threshold, std::vector<Object>& objects)
{
    static float anchors[18] = {10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326};
    //static float anchor_p16[6] = {30, 61, 62, 45, 59, 119};
    //static float anchor_p32[6] = {116, 90, 156, 198, 373, 326};
    int anchor_num = 3;
    int feat_w = 640 / stride;
    int feat_h = 640 / stride;
    int cls_num = 80;
    int anchor_group = 0;
    if(stride == 8)
        anchor_group = 1;
    if(stride == 16)
        anchor_group = 2;
    if(stride == 32)
        anchor_group = 3;
    for (int h = 0; h <= feat_h - 1; h++)
    {
        for (int w = 0; w <= feat_w - 1; w++)
        {
            for (int a = 0; a <= anchor_num - 1; a++)
            {
                //process cls score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int s = 0; s <= cls_num - 1; s++)
                {
                    float score = feat[a * feat_w * feat_h * 85 + h * feat_w * 85 + w * 85 + s + 5];
                    if(score > class_score)
                    {
                        class_index = s;
                        class_score = score;
                    }
                }
                //process box score
                float box_score = feat[a * feat_w * feat_h * 85 + (h * feat_w) * 85 + w * 85 + 4];
                float final_score = sigmoid(box_score ) * sigmoid(class_score);
                if (final_score >= prob_threshold)
                {
                    int loc_idx = a * feat_h * feat_w * 85 + h * feat_w * 85 + w * 85 ;
                    float dx = sigmoid(feat[loc_idx + 0]);
                    float dy = sigmoid(feat[loc_idx + 1]);
                    float dw = sigmoid(feat[loc_idx + 2]);
                    float dh = sigmoid(feat[loc_idx + 3]);
                    float pred_cx = (dx * 2.0f - 0.5f + w) * stride;
                    float pred_cy = (dy * 2.0f - 0.5f + h) * stride;
                    //float pred_w = pow(dw * 2.f, 2) * anchors[a * 2 + 0];
                    //float pred_h = pow(dh * 2.f, 2) * anchors[a * 2 + 1];
                    float anchor_w = anchors[(anchor_group - 1) * 6 + a * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + a * 2 + 1];
                    float pred_w = dw * dw * 4.0f * anchor_w;
                    float pred_h = dh * dh * 4.0f * anchor_h;
                    float x0 = pred_cx - pred_w * 0.5f;
                    float y0 = pred_cy - pred_h * 0.5f;
                    float x1 = pred_cx + pred_w * 0.5f;
                    float y1 = pred_cy + pred_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    objects.push_back(obj);
                }
            }
        }
    }
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0));
    }

    cv::imwrite("yolov5_out.jpg", image);
}

void show_usage()
{
    fprintf(
        stderr,
        "[Usage]:  [-h]\n    [-m model_file] [-i image_file] [-r repeat_count] [-t thread_count] \n");
}

int main(int argc, char* argv[])
{
    const char* model_file = nullptr;
    const char* image_file = nullptr;
    int img_h = 640;
    int img_w = 640;
    int img_c = 3;
    const float mean[3] = {0, 0, 0};
    const float scale[] = {0.003921, 0.003921, 0.003921};

    int repeat_count = 1;
    int num_thread = 1;

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
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
                repeat_count = std::strtoul(optarg, nullptr, 10);
                break;
            case 't':
                num_thread = std::strtoul(optarg, nullptr, 10);
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    /* check files */
    if (nullptr == model_file)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (nullptr == image_file)
    {
        fprintf(stderr, "Error: Image file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file) || !check_file_exist(image_file))
        return -1;

    cv::Mat m = cv::imread(image_file, 1);
    if (m.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", image_file);
        return -1;
    }

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    if (init_tengine() != 0)
    {
        fprintf(stderr, "Initial tengine failed.\n");
        return -1;
    }
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (graph == nullptr)
    {
        fprintf(stderr, "Create graph failed.\n");
        fprintf(stderr, "errno: %d \n", get_tengine_errno());
        return -1;
    }

    int img_size = img_h * img_w * img_c;
    int dims[] = {1, 12, int(img_h / 2), int(img_w / 2)};
    float* input_data = ( float* )malloc(img_size * sizeof(float));

    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }

    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }

    if (set_tensor_buffer(input_tensor, input_data, img_size * 4) < 0)
    {
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }

    /* prerun graph, set work options(num_thread, cluster, precision) */
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }

    /* prepare process input data, set the data mem to input tensor */
    get_input_data_focas(image_file, input_data, img_h, img_w, mean, scale);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < 1; i++)
    {
        double start = get_current_time();
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }
        double end = get_current_time();
        double cur = end - start;
        total_time += cur;
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", 1, 1,
            total_time, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    // 179: 1, 3, 20, 20, 85
    // 178: 1, 3, 40, 40, 85
    // 177: 1, 3, 80, 80, 85
    tensor_t p8_output = get_graph_output_tensor(graph, 177, 0);
    tensor_t p16_output = get_graph_output_tensor(graph, 178, 0);
    tensor_t p32_output = get_graph_output_tensor(graph, 179, 0);

    float* p8_data = ( float*)get_tensor_buffer(p8_output);
    float* p16_data = ( float*)get_tensor_buffer(p16_output);
    float* p32_data = ( float*)get_tensor_buffer(p32_output);

    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;

    std::vector<Object> proposals;
    std::vector<Object> objects8;
    std::vector<Object> objects16;
    std::vector<Object> objects32;
    std::vector<Object> objects;

    static float anchor_p8[6] = {10, 13, 16, 30, 33, 23};
    static float anchor_p16[6] = {30, 61, 62, 45, 59, 119};
    static float anchor_p32[6] = {116, 90, 156, 198, 373, 326};

    generate_proposals(32, p32_data, prob_threshold, objects32);
    proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    generate_proposals(16, p16_data, prob_threshold, objects16);
    proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    generate_proposals( 8, p8_data, prob_threshold, objects8);
    proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    image im = imread(image_file);
    int raw_h = im.h;
    int raw_w = im.w;
    float ratio_x = (float)raw_w / img_w;
    float ratio_y = (float)raw_h / img_h;

    int count = picked.size();
    printf("%d\n",count);
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x) * ratio_x;
        float y0 = (objects[i].rect.y) * ratio_y;
        float x1 = (objects[i].rect.x + objects[i].rect.width) * ratio_x;
        float y1 = (objects[i].rect.y + objects[i].rect.height) * ratio_y;

        x0 = std::max(std::min(x0, (float)(raw_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(raw_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(raw_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(raw_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    draw_objects(m, objects);

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}
