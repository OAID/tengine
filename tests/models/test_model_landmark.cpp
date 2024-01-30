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
 * Author: sqfu@openailab.com
 */

#include <iostream>
#include <vector>
#include <functional>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1
#include <cmath>
int float_mismatch(float* current, float* reference, int size)
{
    for (int i = 0; i < size; i++)
    {
        float tmp = fabs(current[i]) - fabs(reference[i]);
        if (fabs(tmp) > 0.0001)
        {
            fprintf(stderr, "test failed, index:%d, a:%f, b:%f\n", i, current[i], reference[i]);
            return -1;
        }
    }
    fprintf(stderr, "test pass\n");
    return 0;
}
void get_input_fp32_data(const char* image_file, float* input_data, int img_h, int img_w, float* mean, float* scale)
{
    image img = imread_process(image_file, img_w, img_h, mean, scale);

    float* image_data = (float*)img.data;

    for (int i = 0; i < img_w * img_h * 3; i++)
        input_data[i] = image_data[i];

    free_image(img);
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file]  [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char model_string[] = "./models/landmark.tmfile";
    char* model_file = model_string;
    int img_h = 144;
    int img_w = 144;
    float mean[3] = {128.f, 128.f, 128.f};
    float scale[3] = {0.0039, 0.0039, 0.0039};

    int res;
    while ((res = getopt(argc, argv, "m:i:r:t:h:")) != -1)
    {
        switch (res)
        {
        case 'm':
            model_file = optarg;
            break;
        case 'r':
            repeat_count = atoi(optarg);
            break;
        case 't':
            num_thread = atoi(optarg);
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
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file))
        return -1;

    /* set runtime options */
    struct options opt;
    opt.num_thread = num_thread;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_FP32;
    opt.affinity = 0;

    /* inital tengine */
    init_tengine();
    fprintf(stderr, "tengine-lite library version: %s\n", get_tengine_version());

    /* create graph, load tengine model xxx.tmfile */
    graph_t graph = create_graph(nullptr, "tengine", model_file);
    if (graph == nullptr)
    {
        std::cout << "Create graph0 failed\n";
        return -1;
    }

    /* set the input shape to initial the graph, and prerun graph to infer shape */
    int img_size = img_h * img_w * 3;
    int dims[] = {1, 3, img_h, img_w}; // nchw
    float* input_data = (float*)malloc(img_size * sizeof(float));

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

    if (set_tensor_buffer(input_tensor, input_data, img_size * sizeof(float)) < 0)
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
    std::string model_name = "landmark";
    // get_input_fp32_data(image_file, input_data, img_h, img_w, mean, scale);
    std::string input_file = "./data/" + model_name + "_in.bin";
    FILE* fp;
    fp = fopen(input_file.c_str(), "rb");
    if (fread(input_data, sizeof(float), img_size, fp) == 0)
    {
        fprintf(stderr, "read input data file failed!\n");
        return -1;
    }
    fclose(fp);

    /* run graph */
    double min_time = DBL_MAX;
    double max_time = DBL_MIN;
    double total_time = 0.;
    for (int i = 0; i < repeat_count; i++)
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
        if (min_time > cur)
            min_time = cur;
        if (max_time < cur)
            max_time = cur;
    }
    printf("Repeat [%d] min %.3f ms, max %.3f ms, avg %.3f ms\n", repeat_count, min_time, max_time,
           total_time / repeat_count);

    /* get output tensor */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);

    float* output_data = (float*)(get_tensor_buffer(output_tensor));
    int data_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
    // save output_data

    std::string reference_file1 = "./data/" + model_name + "_out.bin";
    std::vector<float> reference_data(data_size);
    FILE* fp1;
    //read
    fp1 = fopen(reference_file1.c_str(), "rb");
    if (fread(reference_data.data(), sizeof(float), data_size, fp1) == 0)
    {
        fprintf(stderr, "read reference data file1 failed!\n");
        return -1;
    }
    fclose(fp1);
    int ret1 = float_mismatch(output_data, reference_data.data(), data_size);

    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return ret1;
}
