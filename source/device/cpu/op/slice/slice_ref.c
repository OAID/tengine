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
 * Author: bhu@openailab.com
 */

#include "slice_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/vector.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>
#include <string.h>
#include <stdbool.h>

struct shape_dim
{
    int dims[4];   // for caffe
    int begins[4]; // for tf
    int sizes[4];  // for tf
};

struct slice_param_ref
{
    int in_shape[4]; // the dim of the input
    int in_shape_3[3];
    int in_shape_2[2];
    struct shape_dim* output_shape; // out shape
    int out_num;
    int dim_num;
    int axis;        // for caffe
    int step;        // for onnx
    float out_scale; // for input tensor int8
    bool iscaffe;
    bool ismxnet;
    bool isonnx;
    int begin;
    int end;
};
static int caffe_run(const int8_t* in_data, int8_t** out_data, int element_size, const struct slice_param_ref* param)
{
    // get the slice param
    int slice_axis = param->axis;
    int num_slices = 1;
    int slice_size = 1;
    const int8_t* input = in_data;
    const int* in_dim = param->in_shape;

    for (int i = 0; i < slice_axis; i++)
    {
        num_slices = num_slices * in_dim[i];
    }
    for (int i = slice_axis + 1; i < param->dim_num; i++)
    {
        slice_size = slice_size * in_dim[i];
    }
    int in_slice = in_dim[slice_axis];
    int slice_index = 0;
    int out_num = param->out_num;
    for (int i = 0; i < out_num; i++)
    {
        int8_t* output = out_data[i];
        int out_slice = param->output_shape[i].dims[slice_axis];
        for (int n = 0; n < num_slices; n++)
        {
            int in_offset = (n * in_slice + slice_index) * slice_size * element_size;
            int out_offset = n * out_slice * slice_size * element_size;
            memcpy(output + out_offset, input + in_offset, (size_t)slice_size * out_slice * element_size);
        }
        slice_index += out_slice;
    }
    return 0;
}
static int tf_run(const int8_t* in_data, int8_t** out_data, int element_size, const struct slice_param_ref* param)
{
    const int8_t* input = in_data;
    int8_t* output = out_data[0];

    const int* begins = param->output_shape[0].begins;
    const int* sizes = param->output_shape[0].sizes;
    int real_dim = param->dim_num;
    const int* in_dim_new = param->in_shape;
    int in_dim_0 = in_dim_new[0];
    int in_dim_1 = in_dim_new[1];
    int in_dim_2 = in_dim_new[2];
    int in_dim_3 = in_dim_new[3];

    int start_dim_0 = (4 - real_dim) > 0 ? 0 : begins[0];
    int stop_dim_0 = ((4 - real_dim) > 0 || sizes[0] == -1) ? in_dim_0 - start_dim_0 : start_dim_0 + sizes[0];
    int start_dim_1 = (3 - real_dim) > 0 ? 0 : begins[1];
    int stop_dim_1 = ((3 - real_dim) > 0 || sizes[1] == -1) ? in_dim_1 - start_dim_1 : start_dim_1 + sizes[1];
    int start_dim_2 = (2 - real_dim) > 0 ? 0 : begins[2];
    int stop_dim_2 = ((2 - real_dim) > 0 || sizes[2] == -1) ? in_dim_2 - start_dim_2 : start_dim_2 + sizes[2];
    int start_dim_3 = (1 - real_dim) > 0 ? 0 : begins[3];
    int stop_dim_3 = ((1 - real_dim) > 0 || sizes[3] == -1) ? in_dim_3 - start_dim_3 : start_dim_3 + sizes[3];

    for (int n = start_dim_0; n < stop_dim_0; ++n)
    {
        for (int i = start_dim_1; i < stop_dim_1; ++i)
        {
            for (int j = start_dim_2; j < stop_dim_2; ++j)
            {
                int len = stop_dim_3 - start_dim_3;
                int input_off = n * in_dim_1 * in_dim_2 * in_dim_3 + i * in_dim_2 * in_dim_3 + j * in_dim_3 + start_dim_3;
                memcpy(output, input + input_off * element_size, (size_t)len * element_size);
                output += len * element_size;
            }
        }
    }
    return 0;
}
static int mxnet_run(const int8_t* in_data, int8_t** out_data, int element_size, const struct slice_param_ref* param)
{
    const int8_t* input = in_data;
    int8_t* output = out_data[0];

    // const int begins = param->begin;
    // const int end = param->end;

    if (param->dim_num == 4)
    {
        const int* in_dim_new = param->in_shape;

        int in_dim_1 = in_dim_new[1];
        int in_dim_2 = in_dim_new[2];
        int in_dim_3 = in_dim_new[3];
        int start_0 = (param->axis == 0) ? param->begin : 0;
        int start_1 = (param->axis == 1) ? param->begin : 0;
        int start_2 = (param->axis == 2) ? param->begin : 0;
        int start_3 = (param->axis == 3) ? param->begin : 0;
        int stop_0 = (param->axis == 0) ? param->end : param->in_shape[0];
        int stop_1 = (param->axis == 1) ? param->end : param->in_shape[1];
        int stop_2 = (param->axis == 2) ? param->end : param->in_shape[2];
        int stop_3 = (param->axis == 3) ? param->end : param->in_shape[3];

        for (int n = start_0; n < stop_0; ++n)
        {
            for (int i = start_1; i < stop_1; ++i)
            {
                for (int j = start_2; j < stop_2; ++j)
                {
                    int len = start_3 - stop_3;
                    int input_off = n * in_dim_1 * in_dim_2 * in_dim_3 + i * in_dim_2 * in_dim_3 + j * in_dim_3 + start_3;
                    memcpy(output, input + input_off * element_size, (size_t)len * element_size);
                    output += len * element_size;
                }
            }
        }
    }
    else if (param->dim_num == 3)
    {
        const int* in_dim_new = param->in_shape_3;
        // int in_dim_0 = in_dim_new[0];
        int in_dim_1 = in_dim_new[1];
        int in_dim_2 = in_dim_new[2];
        // int in_dim_3 = in_dim_new[3];
        int start_0 = (param->axis == 0) ? param->begin : 0;
        int start_1 = (param->axis == 1) ? param->begin : 0;
        int start_2 = (param->axis == 2) ? param->begin : 0;

        // int start_3=(param->axis==3)? param->begin:0;
        int stop_0 = (param->axis == 0) ? param->end : param->in_shape_3[0];
        int stop_1 = (param->axis == 1) ? param->end : param->in_shape_3[1];
        int stop_2 = (param->axis == 2) ? param->end : param->in_shape_3[2];
        // int stop_3=(param->axis==3)? param->end:param->in_shape[3];

        for (int n = start_0; n < stop_0; ++n)
        {
            for (int i = start_1; i < stop_1; ++i)
            {
                int len = stop_2 - start_2;
                int input_off = n * in_dim_1 * in_dim_2 + i * in_dim_2 + start_2;
                memcpy(output, input + input_off * element_size, (size_t)len * element_size);
                output += len * element_size;
            }
        }
    }
    else if (param->dim_num == 2)
    {
        const int* in_dim_new = param->in_shape_2;
        // int in_dim_0 = in_dim_new[0];
        int in_dim_1 = in_dim_new[1];
        // int in_dim_3 = in_dim_new[3];
        int start_0 = (param->axis == 0) ? param->begin : 0;
        int start_1 = (param->axis == 1) ? param->begin : 0;

        // int start_3=(param->axis==3)? param->begin:0;
        int stop_0 = (param->axis == 0) ? param->end : param->in_shape_2[0];
        int stop_1 = (param->axis == 1) ? param->end : param->in_shape_2[1];
        // int stop_3=(param->axis==3)? param->end:param->in_shape[3];

        for (int n = start_0; n < stop_0; ++n)
        {
            int len = stop_1 - start_0;
            int input_off = n * in_dim_1 + start_1;
            memcpy(output, input + input_off * element_size, (size_t)len * element_size);
            output += len * element_size;
        }
    }

    return 0;
}
static int onnx_run(const int8_t* in_data, int8_t** out_data, int element_size, const struct slice_param_ref* param)
{
    const int8_t* input = in_data;
    int8_t* output = out_data[0];

    // const int begins = param->begin;
    // const int end = param->end;

    if (param->dim_num == 4)
    {
        const int* in_dim_new = param->in_shape;

        int in_dim_1 = in_dim_new[1];
        int in_dim_2 = in_dim_new[2];
        int in_dim_3 = in_dim_new[3];
        int start_0 = (param->axis == 0) ? param->begin : 0;
        int start_1 = (param->axis == 1) ? param->begin : 0;
        int start_2 = (param->axis == 2) ? param->begin : 0;
        int start_3 = (param->axis == 3) ? param->begin : 0;
        int stop_0 = (param->axis == 0) ? (param->end > 0 ? param->end : param->in_shape[0] + param->end) : param->in_shape[0];
        int stop_1 = (param->axis == 1) ? (param->end > 0 ? param->end : param->in_shape[1] + param->end) : param->in_shape[1];
        int stop_2 = (param->axis == 2) ? (param->end > 0 ? param->end : param->in_shape[2] + param->end) : param->in_shape[2];
        int stop_3 = (param->axis == 3) ? (param->end > 0 ? param->end : param->in_shape[3] + param->end) : param->in_shape[3];

        if (param->step > 1)
        {
            int step_0 = (param->axis == 0) ? param->step : 1;
            int step_1 = (param->axis == 1) ? param->step : 1;
            int step_2 = (param->axis == 2) ? param->step : 1;
            int step_3 = (param->axis == 3) ? param->step : 1;
            for (int n = start_0; n < stop_0; n = n + step_0)
            {
                for (int i = start_1; i < stop_1; i = i + step_1)
                {
                    for (int j = start_2; j < stop_2; j = j + step_2)
                    {
                        for (int k = start_3; k < stop_3; k = k + step_3)
                        {
                            int input_index = n * in_dim_1 * in_dim_2 * in_dim_3 + i * in_dim_2 * in_dim_3 + j * in_dim_3 + k;
                            memcpy(output, input + input_index * element_size, element_size);
                            output += element_size;
                        }
                    }
                }
            }

            return 0;
        }

        for (int n = start_0; n < stop_0; ++n)
        {
            for (int i = start_1; i < stop_1; ++i)
            {
                for (int j = start_2; j < stop_2; ++j)
                {
                    int len = stop_3 - start_3;
                    int input_off = n * in_dim_1 * in_dim_2 * in_dim_3 + i * in_dim_2 * in_dim_3 + j * in_dim_3 + start_3;
                    memcpy(output, input + input_off * element_size, (size_t)len * element_size);
                    output += len * element_size;
                }
            }
        }
    }
    else if (param->dim_num == 3)
    {
        const int* in_dim_new = param->in_shape_3;
        // int in_dim_0 = in_dim_new[0];
        int in_dim_1 = in_dim_new[1];
        int in_dim_2 = in_dim_new[2];
        // int in_dim_3 = in_dim_new[3];
        int start_0 = (param->axis == 0) ? param->begin : 0;
        int start_1 = (param->axis == 1) ? param->begin : 0;
        int start_2 = (param->axis == 2) ? param->begin : 0;

        // int start_3=(param->axis==3)? param->begin:0;
        int stop_0 = (param->axis == 0) ? (param->end > 0 ? param->end : param->in_shape_3[0] + param->end) : param->in_shape_3[0];
        int stop_1 = (param->axis == 1) ? (param->end > 0 ? param->end : param->in_shape_3[1] + param->end) : param->in_shape_3[1];
        int stop_2 = (param->axis == 2) ? (param->end > 0 ? param->end : param->in_shape_3[2] + param->end) : param->in_shape_3[2];
        // int stop_3=(param->axis==3)? param->end:param->in_shape[3];

        for (int n = start_0; n < stop_0; ++n)
        {
            for (int i = start_1; i < stop_1; ++i)
            {
                int len = stop_2 - start_2;
                int input_off = n * in_dim_1 * in_dim_2 + i * in_dim_2 + start_2;
                memcpy(output, input + input_off * element_size, (size_t)len * element_size);
                output += len * element_size;
            }
        }
    }
    else if (param->dim_num == 2)
    {
        const int* in_dim_new = param->in_shape_2;
        // int in_dim_0 = in_dim_new[0];
        int in_dim_1 = in_dim_new[1];
        // int in_dim_3 = in_dim_new[3];
        int start_0 = (param->axis == 0) ? param->begin : 0;
        int start_1 = (param->axis == 1) ? param->begin : 0;

        // int start_3=(param->axis==3)? param->begin:0;
        int stop_0 = (param->axis == 0) ? (param->end > 0 ? param->end : param->in_shape_2[0] + param->end) : param->in_shape_2[0];
        int stop_1 = (param->axis == 1) ? (param->end > 0 ? param->end : param->in_shape_2[1] + param->end) : param->in_shape_2[1];
        // int stop_3=(param->axis==3)? param->end:param->in_shape[3];

        for (int n = start_0; n < stop_0; ++n)
        {
            int len = stop_1 - start_0;
            int input_off = n * in_dim_1 + start_1;
            memcpy(output, input + input_off * element_size, (size_t)len * element_size);
            output += len * element_size;
        }
    }

    return 0;
}
static int ref_slice_common(const int8_t* in_data, int8_t** out_data, int element_size,
                            const struct slice_param_ref* param)
{
    if (param->iscaffe)
        return caffe_run(in_data, out_data, element_size, param);
    else if (param->ismxnet)
        return mxnet_run(in_data, out_data, element_size, param);
    else if (param->isonnx)
        return onnx_run(in_data, out_data, element_size, param);
    else
        return tf_run(in_data, out_data, element_size, param);
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
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct slice_param_ref op_param;
    slice_param_t* _param = (struct slice_param*)(ir_node->op.param_mem);

    int out_num = exec_node->output_num;

    struct shape_dim sd[MAX_SHAPE_DIM_NUM * 2];
    int8_t** out_data_ptrs = (int8_t**)sys_malloc(out_num * sizeof(int8_t*));
    if (out_data_ptrs == NULL)
    {
        return -1;
    }

    op_param.axis = _param->axis;
    op_param.output_shape = sd;
    op_param.out_num = out_num;
    op_param.dim_num = (int)(input_tensor->dim_num);
    op_param.iscaffe = _param->iscaffe;
    op_param.ismxnet = _param->ismxnet;
    op_param.isonnx = _param->isonnx;

    int8_t* input = (int8_t*)input_tensor->data;
    unsigned int mem_size = input_tensor->elem_size;

    if (op_param.iscaffe)
    {
        // set the input dim and output dim
        for (int i = 0; i < op_param.dim_num; i++)
        {
            op_param.in_shape[i] = input_tensor->dims[i];
        }
        // set the output
        for (int i = 0; i < op_param.out_num; ++i)
        {
            struct tensor* out_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[i]);
            for (int j = 0; j < op_param.dim_num; ++j)
            {
                op_param.output_shape[i].dims[j] = out_tensor->dims[j];
            }
            out_data_ptrs[i] = (int8_t*)out_tensor->data;
        }
    }
    else if (op_param.ismxnet || op_param.isonnx)
    {
        op_param.begin = _param->begin;
        op_param.end = _param->end;
        op_param.axis = _param->axis;
        op_param.step = _param->step;
        op_param.dim_num = input_tensor->dim_num;
        for (unsigned int idx = 0; idx < input_tensor->dim_num; idx++)
        {
            if (input_tensor->dim_num == 4)
            {
                op_param.in_shape[idx] = input_tensor->dims[idx];
            }
            else if (input_tensor->dim_num == 3)
            {
                op_param.in_shape_3[idx] = input_tensor->dims[idx];
            }
            else if (input_tensor->dim_num == 2)
            {
                op_param.in_shape_2[idx] = input_tensor->dims[idx];
            }
        }
        struct tensor* out_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
        out_data_ptrs[0] = (int8_t*)out_tensor->data;

        if (input_tensor->dims[0] == out_tensor->dims[0] && input_tensor->dims[1] == out_tensor->dims[1] && input_tensor->dims[2] == out_tensor->dims[2] && input_tensor->dims[3] == out_tensor->dims[3])
        {
            memcpy((void*)(out_data_ptrs[0]), (void*)input, mem_size * input_tensor->elem_num);
            sys_free(out_data_ptrs);
            return true;
        }
    }
    else // For tensorflow, there is only one output tensor
    {
        int maxdim = 4;
        int real_dim = op_param.dim_num;
        int dim_idx = 0;
        for (int idx = 0; idx < maxdim; idx++)
        {
            if (maxdim - idx > real_dim)
            {
                op_param.output_shape[0].begins[idx] = 0;
                op_param.output_shape[0].sizes[idx] = 1;
                op_param.in_shape[idx] = 1;
            }
            else
            {
                op_param.output_shape[0].begins[idx] = *(int*)get_vector_data(_param->begin_, dim_idx);
                op_param.output_shape[0].sizes[idx] = *(int*)get_vector_data(_param->size_, dim_idx);
                op_param.in_shape[idx] = input_tensor->dims[dim_idx];
                dim_idx++;
            }
        }
        struct tensor* out_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
        out_data_ptrs[0] = (int8_t*)out_tensor->data;
    }

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_slice_common(input, out_data_ptrs, sizeof(float), &op_param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8) // ugly implement, need to refactor !
    {
        /* dequant to fp32 */
        uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
        uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
        float input_scale = input_tensor->scale;
        float output_scale = output_tensor->scale;
        int32_t input_zero = input_tensor->zero_point;
        int32_t output_zero = output_tensor->zero_point;

        float* input_fp32 = (float*)sys_malloc(input_tensor->elem_num * sizeof(float));
        float* output_fp32 = (float*)sys_malloc(output_tensor->elem_num * sizeof(float));
        out_data_ptrs[0] = (int8_t*)output_fp32;

        for (int i = 0; i < input_tensor->elem_num; i++)
        {
            input_fp32[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
        }

        ret = ref_slice_common((int8_t*)input_fp32, out_data_ptrs, sizeof(float), &op_param);

        /* quant to uint8 */
        for (int i = 0; i < output_tensor->elem_num; i++)
        {
            int udata = round(output_fp32[i] / output_scale + output_zero);
            if (udata > 255)
                udata = 255;
            else if (udata < 0)
                udata = 0;
            output_uint8[i] = udata;
        }

        free(input_fp32);
        free(output_fp32);
    }

    sys_free(out_data_ptrs);
    if (ret < 0)
        return -1;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops slice_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_slice_ref_op()
{
    return register_builtin_node_ops(OP_SLICE, &slice_node_ops);
}

int unregister_slice_ref_op()
{
    return unregister_builtin_node_ops(OP_SLICE, &slice_node_ops);
}
