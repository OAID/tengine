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


#include "test_op.h"
#include "operator/prototype/pooling_param.h"


int create_test_pool_node(graph_t graph, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w)
{
    (void)layout; (void)n; (void)c; (void)h; (void)w;

    /* create the test node */
    struct node* test_node = (struct node* )create_graph_node(graph, node_name, "Pooling");

    tensor_t input_tensor = get_graph_tensor(graph, input_name);

    if(NULL == input_tensor)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    /* input tensors of test node */
    set_node_input_tensor(test_node, 0, input_tensor);

    /* output tensors of test node */
    tensor_t output_tensor = create_graph_tensor(graph, node_name, data_type);
    set_node_output_tensor(test_node, 0, output_tensor, TENSOR_TYPE_VAR);

    /* set params */
    struct pool_param* pool_param = ( struct pool_param* )(struct node* )test_node->op.param_mem;

    pool_param->pool_method = POOL_MAX;
    pool_param->global = 0;
    pool_param->kernel_h = 3;
    pool_param->kernel_w = 3;
    pool_param->stride_h = 2;
    pool_param->stride_w = 2;
    pool_param->pad_h0 = 0;
    pool_param->pad_h1 = 0;
    pool_param->pad_w0 = 0;
    pool_param->pad_w1 = 0;
    pool_param->pad_h0_org = 0;
    pool_param->pad_h1_org = 0;
    pool_param->pad_w0_org = 0;
    pool_param->pad_w1_org = 0;
    pool_param->caffe_flavor = 0;
    pool_param->funct = NULL;

    return 0;
}

float reference_out[3] = {5.f, 5.f, 5.f};


float input_scale = 0.039216f;
int input_zero_point = 255;
float output_scale = 0.039216f;
int output_zero_point = 255;

int main(int argc, char* argv[])
{
    int n = 1, c = 3, h = 4, w = 5;
    const char* test_node_name = "pooling";
    int data_type = TENGINE_DT_INT8;
    int layout = TENGINE_LAYOUT_NCHW;

    // init
    int ret = test_graph_init();
    if (0 != ret)
        fprintf(stderr, "Tengine init failed.\n");

    // create
    graph_t graph = create_opendla_test_graph(test_node_name, data_type, layout, n, c, h, w, &create_test_pool_node);
    if(NULL == graph)
        return -1;

    // set quantize params
    tensor_t input_tesnor = get_graph_input_tensor(graph, 0, 0);
    tensor_t output_tesnor = get_graph_output_tensor(graph, 0, 0);
    set_tensor_quant_param(input_tesnor, &input_scale, &input_zero_point, 1);
    set_tensor_quant_param(output_tesnor, &output_scale, &output_zero_point, 1);

    // set input data
    fill_input_int8_tensor_by_index(graph, 0, 0, 1.0f);
    // graph run
    ret = test_graph_run(graph);
    if (0 != ret)
    {
        fprintf(stderr, "Run graph error. ERRNO: %d.\n", ret);
        test_graph_release(graph);
        return -1;
    }

    // get output and dequant
    struct tensor* output_tensor = (struct tensor*)get_graph_output_tensor(graph, 0, 0);
    int8_t* output_int8 = ( int8_t* )output_tensor->data;
    int output_size = output_tensor->elem_num;
    int out_c = output_tensor->dims[1];
    int cstep = output_tensor->dims[2] * output_tensor->dims[3];

    get_tensor_quant_param(output_tensor, &output_scale, &output_zero_point, 1);
    float* output_data = ( float* )malloc(output_size * sizeof(float));
    for (int i = 0; i < output_size; i++)
        output_data[i] = (( float )output_int8[i] - ( float )output_zero_point) * output_scale;

    // check the result
    ret = 0;
    for (int i = 0; i< out_c; i++)
    {
        float* output_value =  (float *)output_data + i * cstep;
        for (int j = 0; j < cstep; j++)
        {
            fprintf(stdout, "index:%d, a:%f, b:%f\n", j, output_value[j], reference_out[i]);
            if (fabsf(output_value[j] - reference_out[i]) > 0.01)
            {
                fprintf(stderr, "index:%d, a:%f, b:%f\n", j, output_value[j], reference_out[i]);
                ret = -1;
            }
        }
    }

    if (ret == 0)
        fprintf(stderr, "test pass.\n");
    else
        fprintf(stderr, "test failed.\n");

    // exit
    test_graph_release(graph);

    return ret;
}
