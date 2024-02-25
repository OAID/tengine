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

#include "topkv2_param.h"

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

struct topkv2_param_ref
{
    int k;
    int row_size;
    int num_rows;
};

static void swap_fp32(float* p, float* q)
{
    float buf;
    buf = *p;
    *p = *q;
    *q = buf;
    return;
}
static void swap_int(int* p, int* q)
{
    int buf;
    buf = *p;
    *p = *q;
    *q = buf;
    return;
}
static void quick_sort_fp32(float* a, int low, int high, int* indexv)
{
    int i = low;
    int j = high;
    float key = a[low];

    if (low >= high)
    {
        return;
    }

    while (low < high)
    {
        while (low < high && key >= a[high])
        {
            --high;
        }

        if (key < a[high])
        {
            swap_fp32(&a[low], &a[high]);
            swap_int(&indexv[low], &indexv[high]);

            ++low;
        }
        while (low < high && key <= a[low])
        {
            ++low;
        }
        if (key > a[low])
        {
            swap_fp32(&a[low], &a[high]);
            swap_int(&indexv[low], &indexv[high]);

            --high;
        }
    }
    quick_sort_fp32(a, i, low - 1, indexv);
    quick_sort_fp32(a, low + 1, j, indexv);
}

static int ref_topkv2_fp32(float* in_data, float* out_data, int* out_index, struct topkv2_param_ref* param)
{
    int k = param->k;

    int row_size = param->row_size;
    int num_rows = param->num_rows;
    int* index = (int*)sys_malloc(row_size * sizeof(int));

    for (int i = 0; i < num_rows; ++i)
    {
        int start = i * row_size;
        for (int j = 0; j < row_size; ++j)
            index[j] = j;

        quick_sort_fp32(&in_data[start], 0, row_size - 1, index);
        memcpy(&out_data[i * k], &in_data[start], k * sizeof(float));
        memcpy(&out_index[i * k], index, k * sizeof(float));
        sys_free(index);
    }

    return 0;
}

static int ref_topkv2_uint8(struct tensor* input_tensor, struct tensor* output_tensor, int* out_index, struct topkv2_param_ref* param)
{
    /* dequant */
    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* in_data = (float*)sys_malloc(input_size * sizeof(float));
    float* out_data = (float*)sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        in_data[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    int k = param->k;
    int row_size = param->row_size;
    int num_rows = param->num_rows;
    int* index = (int*)sys_malloc(row_size * sizeof(int));

    for (int i = 0; i < num_rows; ++i)
    {
        int start = i * row_size;
        for (int j = 0; j < row_size; ++j)
            index[j] = j;

        quick_sort_fp32(&in_data[start], 0, row_size - 1, index);

        memcpy(&out_data[i * k], &in_data[start], k * sizeof(float));
        memcpy(&out_index[i * k], index, k * sizeof(float));
        sys_free(index);
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(out_data[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(in_data);
    sys_free(out_data);

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
    struct topkv2_param* _param = (struct topkv2_param*)(ir_node->op.param_mem);
    struct tensor* input_tensor;
    int out_nums = ir_node->output_num;
    struct topkv2_priv_info* topkv2_priv_info = (struct topkv2_priv_info*)exec_node->ops_priv;
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct tensor* output_tensor_1 = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[1]);
    int dims_len = input_tensor->dim_num;
    int num_rows = 1;
    for (int i = 0; i < dims_len - 1; ++i)
    {
        num_rows *= input_tensor->dims[i];
    }
    struct topkv2_param_ref op_param;
    op_param.k = _param->k;
    op_param.row_size = input_tensor->dims[dims_len - 1];
    op_param.num_rows = num_rows;
    float* input = (float*)input_tensor->data;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_topkv2_fp32(input, (float*)output_tensor->data, (int*)output_tensor_1->data, &op_param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_topkv2_uint8(input_tensor, output_tensor, (int*)output_tensor_1->data, &op_param);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_topkv2_ref_op()
{
    return register_builtin_node_ops(OP_TOPKV2, &hcl_node_ops);
}

int unregister_topkv2_ref_op()
{
    return unregister_builtin_node_ops(OP_TOPKV2, &hcl_node_ops);
}
