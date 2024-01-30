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
 *
 * original pretrained: https://github.com/meijieru/crnn.pytorch
 */

#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <fstream>
#include <vector>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

#define DEFAULT_REPEAT_COUNT 1
#define DEFAULT_THREAD_COUNT 1

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

std::string read_txt(const std::string& filename, int line)
{
    std::ifstream fin;
    fin.open(filename, std::ios::in);
    std::string strVec[5530];
    int i = 0;
    while (!fin.eof())
    {
        std::string inbuf;
        getline(fin, inbuf, '\n');
        strVec[i] = inbuf;
        i = i + 1;
    }
    return strVec[line - 1];
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file]  [-r repeat_count] [-t thread_count]\n");
}

int main(int argc, char* argv[])
{
    int repeat_count = DEFAULT_REPEAT_COUNT;
    int num_thread = DEFAULT_THREAD_COUNT;
    char model_string[] = "./models/crnn_lite_dense.tmfile";
    char* model_file = model_string;
    char* image_file = nullptr;
    char* label_file = nullptr;
    int img_h = 32;
    int img_w = 277;
    float mean[3] = {127.5, 127.5, 127.5};
    float scale[3] = {0.007843, 0.007843, 0.007843};

    int res;
    while ((res = getopt(argc, argv, "m:i:l:r:t:h:")) != -1)
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
    std::string model_name = "crnn_lite_dense";
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
        return -1;
    }

    int img_size = img_h * img_w * 1;
    int dims[] = {1, 1, img_h, img_w};
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
    double min_time = __DBL_MAX__;
    double max_time = -__DBL_MAX__;
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
        min_time = std::min(min_time, cur);
        max_time = std::max(max_time, cur);
    }
    fprintf(stderr, "Repeat %d times, thread %d, avg time %.2f ms, max_time %.2f ms, min_time %.2f ms\n", repeat_count,
            num_thread, total_time / repeat_count, max_time, min_time);
    fprintf(stderr, "--------------------------------------\n");

    /* process the crnn result */
    tensor_t output_tensor = get_graph_output_tensor(graph, 0, 0);
    float* ocr_data = (float*)get_tensor_buffer(output_tensor);

    /* check the result */
    int output_size = get_tensor_buffer_size(output_tensor) / sizeof(float);
    std::string reference_file = "./data/" + model_name + "_out.bin";
    std::vector<float> reference_data(output_size);
    FILE* fp1;
    fp1 = fopen(reference_file.c_str(), "rb");
    if (fread(reference_data.data(), sizeof(float), output_size, fp1) == 0)
    {
        fprintf(stderr, "read reference data file failed!\n");
        return -1;
    }
    fclose(fp1);

    int ret = float_mismatch(ocr_data, reference_data.data(), output_size);

    //    process_crnn_result(ocr_data, label_file);

    free(input_data);
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();

    return ret;
}
