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

#include "split_param.h"

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

int ref_split_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct split_param* split_param, int* slice_index, int num_slices, int slice_size, int in_slice, int slice_axis)
{
    float* input_data = (float*)input_tensor->data;
    float* output_data = (float*)output_tensor->data;

    if (split_param->is_caffe)
    {
        memcpy(output_data, input_data, input_tensor->elem_num * sizeof(float));
    }
    else
    {
        int out_slice = 0;

        out_slice = output_tensor->dims[slice_axis];

        for (int n = 0; n < num_slices; n++)
        {
            int in_offset = (n * in_slice + *slice_index) * slice_size;
            int out_offset = n * out_slice * slice_size;
            memcpy(output_data + out_offset, input_data + in_offset, (size_t)slice_size * out_slice * sizeof(float));
        }

        *slice_index += out_slice;
    }

    return 0;
}

int ref_split_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct split_param* split_param, int* slice_index, int num_slices, int slice_size, int in_slice, int slice_axis)
{
    uint8_t* input_data = (uint8_t*)input_tensor->data;
    uint8_t* output_data = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    float rescale = input_scale / output_scale;

    if (split_param->is_caffe)
    {
        memcpy(output_data, input_data, input_tensor->elem_num * sizeof(uint8_t));
    }
    else
    {
        int out_slice = 0;

        out_slice = output_tensor->dims[slice_axis];

        for (int n = 0; n < num_slices; n++)
        {
            int in_offset = (n * in_slice + *slice_index) * slice_size;
            int out_offset = n * out_slice * slice_size;
            for (size_t i = 0; i < slice_size * out_slice * sizeof(uint8_t); i++)
            {
                int udata = roundf((input_data[in_offset + i] - input_zero) * rescale + output_zero);
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;
                output_data[i] = udata;
            }
        }

        *slice_index += out_slice;
    }

    return 0;
}

int ref_split_int8(struct tensor* input_tensor, struct tensor* output_tensor, struct split_param* split_param, int* slice_index, int num_slices, int slice_size, int in_slice, int slice_axis)
{
    int8_t* input_data = (int8_t*)input_tensor->data;
    int8_t* output_data = (int8_t*)output_tensor->data;

    if (split_param->is_caffe)
    {
        memcpy(output_data, input_data, input_tensor->elem_num * sizeof(int8_t));
    }
    else
    {
        int out_slice = 0;

        out_slice = output_tensor->dims[slice_axis];

        for (int n = 0; n < num_slices; n++)
        {
            int in_offset = (n * in_slice + *slice_index) * slice_size;
            int out_offset = n * out_slice * slice_size;
            memcpy(output_data + out_offset, input_data + in_offset, (size_t)slice_size * out_slice * sizeof(int8_t));
        }

        *slice_index += out_slice;
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

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    // struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    // output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct split_param* split_param = (struct split_param*)ir_node->op.param_mem;

    /* the follow codes need to be checked ! */
    /* maybe int8 need dequant and quant */
    int slice_axis = split_param->axis;
    int num_slices = 1;
    int slice_size = 1;

    for (int i = 0; i < slice_axis; i++)
        num_slices = num_slices * input_tensor->dims[i];

    for (int i = slice_axis + 1; i < input_tensor->dim_num; i++)
        slice_size = slice_size * input_tensor->dims[i];

    int in_slice = input_tensor->dims[slice_axis];
    int slice_index = 0;
    int out_num = ir_node->output_num;

    int ret = -1;
    for (int i = 0; i < out_num; i++)
    {
        struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[i]);

        if (input_tensor->data_type == TENGINE_DT_FP32)
            ret = ref_split_fp32(input_tensor, output_tensor, split_param, &slice_index, num_slices, slice_size, in_slice, slice_axis);
        else if (input_tensor->data_type == TENGINE_DT_INT8)
            ret = ref_split_int8(input_tensor, output_tensor, split_param, &slice_index, num_slices, slice_size, in_slice, slice_axis);
        else if (input_tensor->data_type == TENGINE_DT_UINT8)
            ret = ref_split_uint8(input_tensor, output_tensor, split_param, &slice_index, num_slices, slice_size, in_slice, slice_axis);
    }

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_split_ref_op()
{
    return register_builtin_node_ops(OP_SPLIT, &hcl_node_ops);
}

int unregister_split_ref_op()
{
    return unregister_builtin_node_ops(OP_SPLIT, &hcl_node_ops);
}
