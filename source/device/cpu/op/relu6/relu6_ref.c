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
 * Author: qtang@openailab.com
 */

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

int ref_relu6_uint8(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread)
{
    int w = input_tensor->dims[3];
    int h = output_tensor->dims[2];
    int channels = input_tensor->dims[1];
    int batch = input_tensor->dims[0];

    int size = h * w;
    int c_step = h * w;
    int batch_step = c_step * channels;
    int total_size = batch_step * batch;

    // dequant
    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* data_fp32 = (float*)sys_malloc(total_size * sizeof(float));

    for (int i = 0; i < total_size; i++)
        data_fp32[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;

    for (int n = 0; n < batch; n++)
    {
        //#pragma omp parallel for num_threads(num_thread)
        for (int q = 0; q < channels; q++)
        {
            float* src = data_fp32 + batch_step * n + c_step * q;
            float* dst = data_fp32 + batch_step * n + c_step * q;

            for (int i = 0; i < size; i++)
            {
                dst[i] = src[i];
                if (src[i] > 6)
                    dst[i] = 6;
                else if (src[i] < 0)
                    dst[i] = 0;
            }
        }
    }

    // quant
    for (int i = 0; i < total_size; i++)
    {
        int udata = round(data_fp32[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }
    sys_free(data_fp32);
    return 0;
}

int ref_relu6_fp32(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread)
{
    int w = input_tensor->dims[3];
    int h = output_tensor->dims[2];
    int channels = input_tensor->dims[1];
    int size = h * w;
    int c_step = h * w;

    float* input_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

#pragma omp parallel for num_threads(num_thread)
    for (int q = 0; q < channels; q++)
    {
        float* src = input_data + c_step * q;
        float* dst = out_data + c_step * q;

        for (int i = 0; i < size; i++)
        {
            dst[i] = src[i];
            if (dst[i] > 6)
                dst[i] = 6;
            if (dst[i] < 0)
                dst[i] = 0;
        }
    }

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

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_relu6_fp32(input_tensor, output_tensor, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_relu6_uint8(input_tensor, output_tensor, exec_graph->num_thread);

    return ret;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    int ret = set_ir_tensor_shape(output, input->dims, input->dim_num);
    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_relu6_ref_op()
{
    return register_builtin_node_ops(OP_RELU6, &hcl_node_ops);
}

int unregister_relu6_ref_op()
{
    return unregister_builtin_node_ops(OP_RELU6, &hcl_node_ops);
}
