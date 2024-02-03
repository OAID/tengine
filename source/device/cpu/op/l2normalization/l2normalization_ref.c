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
 * Author: bzhang@openailab.com
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

int ref_l2normalization_uint8(struct tensor* input_tensor, struct tensor* output_tensor, int size, int channel_size)
{
    int total_size = input_tensor->elem_num;

    /* dequant */
    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* in_data_fp32 = (float*)malloc(total_size * sizeof(float));

    for (int i = 0; i < total_size; i++)
    {
        in_data_fp32[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    float sq_l2_norm = 0;
    for (int j = 0; j < channel_size; j++)
    {
        const float val = in_data_fp32[j];
        sq_l2_norm += val * val;
    }
    const float l2_norm = sqrt(sq_l2_norm);

    for (int j = 0; j < channel_size; j++)
    {
        float output_fp32 = in_data_fp32[j] / l2_norm;
        int udata = round(output_fp32 / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[j] = udata;
    }
    free(in_data_fp32);
    return 0;
}

int ref_l2normalization_fp32(float* input_data, float* output_data, int size, int channel_size)
{
    float sq_l2_norm = 0;
    for (int j = 0; j < channel_size; j++)
    {
        const float val = input_data[j];
        sq_l2_norm += val * val;
    }
    const float l2_norm = sqrt(sq_l2_norm);
    for (int j = 0; j < channel_size; j++)
    {
        output_data[j] = input_data[j] / l2_norm;
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

    int input_size = 1;
    int channel_size = input_tensor->dims[1];

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_l2normalization_uint8(input_tensor, output_tensor, input_size, channel_size);
    else if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_l2normalization_fp32((float*)input_tensor->data, (float*)output_tensor->data, input_size, channel_size);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

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

int register_l2normalization_ref_op()
{
    return register_builtin_node_ops(OP_L2NORMALIZATION, &hcl_node_ops);
}

int unregister_l2normalization_ref_op()
{
    return unregister_builtin_node_ops(OP_L2NORMALIZATION, &hcl_node_ops);
}
