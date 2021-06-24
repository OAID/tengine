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
* Author: qtang@openailab.com
*/

#include <cstdlib>
#include <cstdio>
#include <vector>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"
#include "json/json.h"
#include "json/json-forwards.h"
#include "jsoncpp.cpp"
#include <string>
#include <fstream>
#include <iostream>
#include "tokenization.h"
#include "tokenization.cpp"

using namespace std;
graph_t graph;
tensor_t unique_ids_raw_output;
tensor_t segment_ids;
tensor_t input_mask;
tensor_t input_ids;

tensor_t unique_ids;
tensor_t unstack_1;
tensor_t unstack_0;
int feature_len;

void _truncate_seq_pair(std::vector<std::string>* tokens_a,
                        std::vector<std::string>* tokens_b,
                        size_t max_length) {
// This is a simple heuristic which will always truncate the longer sequence
// one token at a time. This makes more sense than truncating an equal percent
// of tokens from each, since if one sequence is very short then each token
// that's truncated likely contains more information than a longer sequence.
    while (true) {
        size_t total_length = tokens_a->size() + tokens_b->size();
        if (total_length <= max_length) {
            break;
        }
        if (tokens_a->size() > tokens_b->size()) {
            tokens_a->pop_back();
        } else {
            tokens_b->pop_back();
        }
    }
}

void convert_single_example(cuBERT::FullTokenizer* tokenizer,
                            size_t max_seq_length,
                            const char* text_a, const char* text_b,
                            int *input_ids, int8_t *input_mask, int8_t *segment_ids) {
    std::vector<std::string> tokens_a;
    tokens_a.reserve(max_seq_length);

    std::vector<std::string> tokens_b;
    tokens_b.reserve(max_seq_length);

    tokenizer->tokenize(text_a, &tokens_a, max_seq_length);
    if (text_b != nullptr) {
        tokenizer->tokenize(text_b, &tokens_b, max_seq_length);

        // Modifies `tokens_a` and `tokens_b` in place so that the total
        // length is less than the specified length.
        // Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(&tokens_a, &tokens_b, max_seq_length - 3);
    } else {
        if (tokens_a.size() > max_seq_length - 2) {
            tokens_a.resize(max_seq_length - 2);
        }
    }

    // The convention in BERT is:
    // (a) For sequence pairs:
    //  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    //  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    // (b) For single sequences:
    //  tokens:   [CLS] the dog is hairy . [SEP]
    //  type_ids: 0     0   0   0  0     0 0
    //
    // Where "type_ids" are used to indicate whether this is the first
    // sequence or the second sequence. The embedding vectors for `type=0` and
    // `type=1` were learned during pre-training and are added to the wordpiece
    // embedding vector (and position vector). This is not *strictly* necessary
    // since the [SEP] token unambiguously separates the sequences, but it makes
    // it easier for the model to learn the concept of sequences.
    //
    // For classification tasks, the first vector (corresponding to [CLS]) is
    // used as as the "sentence vector". Note that this only makes sense because
    // the entire model is fine-tuned.
    input_ids[0] = tokenizer->convert_token_to_id("[CLS]");
    segment_ids[0] = 0;
    for (int i = 0; i < tokens_a.size(); ++i) {
        input_ids[i + 1] = tokenizer->convert_token_to_id(tokens_a[i]);
        segment_ids[i + 1] = 0;
    }
    input_ids[tokens_a.size() + 1] = tokenizer->convert_token_to_id("[SEP]");
    segment_ids[tokens_a.size() + 1] = 0;

    if (text_b != nullptr) {
        for (int i = 0; i < tokens_b.size(); ++i) {
            input_ids[i + tokens_a.size() + 2] = tokenizer->convert_token_to_id(tokens_b[i]);
            segment_ids[i + tokens_a.size() + 2] = 1;
        }
        input_ids[tokens_b.size() + tokens_a.size() + 2] = tokenizer->convert_token_to_id("[SEP]");
        segment_ids[tokens_b.size() + tokens_a.size() + 2] = 1;
    }

    size_t len = text_b != nullptr ? tokens_a.size() + tokens_b.size() + 3 : tokens_a.size() + 2;
    std::fill_n(input_mask, len, 1);

    // Zero-pad up to the sequence length.
    std::fill_n(input_ids + len, max_seq_length - len, 0);
    std::fill_n(input_mask + len, max_seq_length - len, 0);
    std::fill_n(segment_ids + len, max_seq_length - len, 0);
}

void* cuBERT_open_tokenizer(const char* vocab_file, int do_lower_case) {
    return new cuBERT::FullTokenizer(vocab_file, do_lower_case);
}

std::vector<std::string> read_squad_examples(const char* input_file)
{
    vector<string> examples;
    Json::Value root;//定义根节点
	Json::Reader reader;
    std::ifstream in;
    in.open ("/home/yicheng/huozhu/inputs.json", ios::binary);//输入json文件的绝对路径
    
	if (!in.is_open())
	{
		cout << "文件打开错误"<<endl;
	}
    if (reader.parse(in, root))
    {
        string x = root["data"]["paragraphs"]["qas"][0]["id"].asString();
        int size = root["data"]["paragraphs"]["qas"].size();
        for (int i=0; i<size; i++){
            examples.push_back(root["data"]["paragraphs"]["qas"][i]["id"].asString());
            examples.push_back(root["data"]["paragraphs"]["qas"][i]["question"].asString());
            examples.push_back(root["data"]["paragraphs"]["context"].asString());
        
        }
        return examples;
    }

}
void init(const char* modelfile)
{
    int dims1[2] = {1, 256};
    int dims2[1] = {1};
    init_tengine();
    fprintf(stderr, "tengine version: %s\n", get_tengine_version());
    graph = create_graph(NULL, "tengine", modelfile);
    if (graph == NULL)
    {
        fprintf(stderr, "grph nullptr\n");
    }
    else
    {
        fprintf(stderr, "success init graph\n");
    }
    unique_ids_raw_output = get_graph_input_tensor(graph, 0, 0);
    segment_ids = get_graph_input_tensor(graph, 1, 0);
    input_mask = get_graph_input_tensor(graph, 2, 0);
    input_ids = get_graph_input_tensor(graph, 3, 0);

    set_tensor_shape(unique_ids_raw_output, dims2, 1);
    set_tensor_shape(segment_ids, dims1, 2);
    set_tensor_shape(input_mask, dims1, 2);
    set_tensor_shape(input_ids, dims1, 2);



    int rc = prerun_graph(graph);
    dump_graph(graph);
    unique_ids = get_graph_output_tensor(graph, 0, 0);
    unstack_1 = get_graph_output_tensor(graph, 1, 0);
    unstack_0 = get_graph_output_tensor(graph, 2, 0);

    //get_tensor_shape(output_tensor, dims, 4);
    //feature_len = dims[1];
    fprintf(stderr, "bert prerun %d\n", rc);
}

int getResult()
{
   
    std::vector<float> input_data1(1,1);
    std::vector<float> input_data2(256,1);
    std::vector<float> input_data3(256,1);
    std::vector<float> input_data4(256,1);
    //get_input_data(imagefile, input_data.data(), height, width, means, scales);
    set_tensor_buffer(unique_ids_raw_output, input_data1.data(), 1 * sizeof(float));
    set_tensor_buffer(segment_ids, input_data2.data(), 256 * sizeof(float));
    set_tensor_buffer(input_mask, input_data3.data(), 256 * sizeof(float));
    set_tensor_buffer(input_ids, input_data4.data(), 256 * sizeof(float));

    //set_graph_layout(graph, 2);

    if (run_graph(graph, 1) < 0)
    {
        fprintf(stderr, "run_graph fail");
        return -1;
    }
    float* data1 = ( float* )get_tensor_buffer(unique_ids);
    float* data2 = ( float* )get_tensor_buffer(unstack_1);
    float* data3 = ( float* )get_tensor_buffer(unstack_0);
    
    printf ("data1: %f\n",*data1);
    printf ("data1: %f\n",data1[1]);
    printf ("data1: %f\n",data1[2]);

    printf ("data2: %f\n",data2[0]);
    printf ("data2: %f\n",data2[1]);
    printf ("data2: %f\n",data2[2]);

    printf ("data3: %f\n",data3[0]);
    printf ("data3: %f\n",data3[1]);
    printf ("data3: %f\n",data3[2]);

}

void release()
{
    release_graph_tensor(unique_ids);
    release_graph_tensor(unstack_1);
    release_graph_tensor(unstack_0);
    release_graph_tensor(unique_ids_raw_output);
    release_graph_tensor(segment_ids);
    release_graph_tensor(input_mask);
    release_graph_tensor(input_ids);
    destroy_graph(graph);
}

void show_usage()
{
    fprintf(stderr, "[Usage]:  [-h]\n    [-m model_file] [-a person_a -b person_b]\n [-t thread_count]\n");
    fprintf(stderr, "\nmobilefacenet example: \n    ./mobilefacenet -m /path/to/mobilenet.tmfile -a "
                    "/path/to/person_a.jpg -b /path/to/person_b.jpg\n");
}

int main(int argc, char* argv[])
{
    char* model_file = NULL;
    char* input_file = NULL;
    char* vocab_file = NULL;
    int max_seq_length = 256;
    int doc_stride = 128;
    int max_query_length = 64;
    int batch_size = 3;
    int n_best_size = 20;
    int max_answer_length = 30;

    int res;
    while ((res = getopt(argc, argv, "m:i:v:h")) != -1)
    {
        switch (res)
        {
            case 'm':
                model_file = optarg;
                break;
            case 'i':
                input_file = optarg;
                break;
            case 'v':
                vocab_file = optarg;
                break;
            case 'h':
                show_usage();
                return 0;
            default:
                break;
        }
    }

    vector<string> examples= read_squad_examples(input_file);
    //printf("%s\n",examples[0]);
    /* check files */

    if (model_file == NULL)
    {
        fprintf(stderr, "Error: Tengine model file not specified!\n");
        show_usage();
        return -1;
    }

    if (!check_file_exist(model_file))
        return -1;
    void* tokenizer = cuBERT_open_tokenizer(vocab_file, 1);


    int input_ids[batch_size * max_seq_length];
    int8_t input_mask[batch_size * max_seq_length];
    int8_t segment_ids[batch_size * max_seq_length];
    for (int batch_idx = 0; batch_idx < 3; ++batch_idx) {
        convert_single_example((cuBERT::FullTokenizer *) tokenizer,
                               max_seq_length,
                               examples[3*batch_idx+1].c_str(),
                               examples[3*batch_idx+2].c_str(),
                               input_ids + max_seq_length * batch_idx,
                               input_mask + max_seq_length * batch_idx,
                               segment_ids + max_seq_length * batch_idx);
    }

    init(model_file);



    getResult();



    release();
    return 0;
}