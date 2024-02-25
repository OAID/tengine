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
 * Author: hhchen@openailab.com
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

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

static int ref_prelu_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* slope_tensor)
{
    int dim_num = input_tensor->dim_num;
    int slope_num = slope_tensor->elem_num;

    float* input_data = (float*)input_tensor->data;
    float* output_data = (float*)output_tensor->data;
    float* slope = (float*)slope_tensor->data;

    if (dim_num == 2)
    {
        int n = input_tensor->dims[0];
        int w = input_tensor->dims[1];

        if (slope_num > 1)
        {
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < w; k++)
                {
                    int offset = i * w + k;
                    output_data[offset] = MAX(input_data[offset], 0) + slope[k] * MIN(input_data[offset], 0.f);
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < w; k++)
                {
                    int offset = i * w + k;
                    output_data[offset] = MAX(input_data[offset], 0) + slope[0] * MIN(input_data[offset], 0.f);
                }
            }
        }
    }

    if (dim_num == 3)
    {
        int n = input_tensor->dims[0];
        int c = input_tensor->dims[1];
        int w = input_tensor->dims[2];

        if (slope_num > 1)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; c++)
                {
                    for (int k = 0; k < w; k++)
                    {
                        int offset = i * c * w + j * w + k;
                        output_data[offset] = MAX(input_data[offset], 0) + slope[j] * MIN(input_data[offset], 0.f);
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    for (int k = 0; k < w; k++)
                    {
                        int offset = i * c * w + j * w + k;
                        output_data[offset] = MAX(input_data[offset], 0) + slope[0] * MIN(input_data[offset], 0.f);
                    }
                }
            }
        }
    }

    if (dim_num == 4)
    {
        int n = input_tensor->dims[0];
        int c = input_tensor->dims[1];
        int h = input_tensor->dims[2];
        int w = input_tensor->dims[3];

        if (slope_num > 1)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    for (int l = 0; l < h; l++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            int offset = i * c * h * w + j * h * w + l * w + k;
                            output_data[offset] = MAX(input_data[offset], 0) + slope[j] * MIN(input_data[offset], 0.f);
                        }
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    for (int l = 0; l < h; l++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            int offset = i * c * h * w + j * h * w + l * w + k;
                            output_data[offset] = MAX(input_data[offset], 0) + slope[0] * MIN(input_data[offset], 0.f);
                        }
                    }
                }
            }
        }
    }

    return 0;
}

static int ref_prelu_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* slope_tensor)
{
    int dim_num = input_tensor->dim_num;
    int slope_num = slope_tensor->elem_num;

    uint8_t* input_data = (uint8_t*)input_tensor->data;
    uint8_t* output_data = (uint8_t*)output_tensor->data;
    fp16_t* slope_fp16 = (fp16_t*)slope_tensor->data;

    /* dequant */
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    uint32_t input_zero = input_tensor->zero_point;
    uint32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;
    int slope_size = slope_tensor->elem_num;

    float* input_fp32 = (float*)sys_malloc(input_size * sizeof(float));
    float* output_fp32 = (float*)sys_malloc(output_size * sizeof(float));
    float* slope_fp32 = (float*)sys_malloc(slope_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input_fp32[i] = ((float)input_data[i] - (float)input_zero) * input_scale;
    }
    for (int i = 0; i < slope_size; i++)
    {
        slope_fp32[i] = fp16_to_fp32(slope_fp16[i]);
    }

    if (dim_num == 2)
    {
        int n = input_tensor->dims[0];
        int w = input_tensor->dims[1];

        if (slope_num > 1)
        {
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < w; k++)
                {
                    int offset = i * w + k;
                    output_fp32[offset] = MAX(input_fp32[offset], 0) + slope_fp32[k] * MIN(input_fp32[offset], 0.f);
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                for (int k = 0; k < w; k++)
                {
                    int offset = i * w + k;
                    output_fp32[offset] = MAX(input_fp32[offset], 0) + slope_fp32[0] * MIN(input_fp32[offset], 0.f);
                }
            }
        }
    }

    if (dim_num == 3)
    {
        int n = input_tensor->dims[0];
        int c = input_tensor->dims[1];
        int w = input_tensor->dims[2];

        if (slope_num > 1)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; c++)
                {
                    for (int k = 0; k < w; k++)
                    {
                        int offset = i * c * w + j * w + k;
                        output_fp32[offset] = MAX(input_fp32[offset], 0) + slope_fp32[j] * MIN(input_fp32[offset], 0.f);
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    for (int k = 0; k < w; k++)
                    {
                        int offset = i * c * w + j * w + k;
                        output_fp32[offset] = MAX(input_fp32[offset], 0) + slope_fp32[0] * MIN(input_fp32[offset], 0.f);
                    }
                }
            }
        }
    }

    if (dim_num == 4)
    {
        int n = input_tensor->dims[0];
        int c = input_tensor->dims[1];
        int h = input_tensor->dims[2];
        int w = input_tensor->dims[3];

        if (slope_num > 1)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    for (int l = 0; l < h; l++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            int offset = i * c * h * w + j * h * w + l * w + k;
                            output_fp32[offset] = MAX(input_fp32[offset], 0) + slope_fp32[j] * MIN(input_fp32[offset], 0.f);
                        }
                    }
                }
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < c; j++)
                {
                    for (int l = 0; l < h; l++)
                    {
                        for (int k = 0; k < w; k++)
                        {
                            int offset = i * c * h * w + j * h * w + l * w + k;
                            output_fp32[offset] = MAX(input_fp32[offset], 0) + slope_fp32[0] * MIN(input_fp32[offset], 0.f);
                        }
                    }
                }
            }
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(output_fp32[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_data[i] = udata;
    }

    sys_free(input_fp32);
    sys_free(output_fp32);
    sys_free(slope_fp32);

    return 0;
}

static int ref_prelu_int8(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* slope_tensor)
{
    int dim0 = input_tensor->dims[0];
    int dim1 = input_tensor->dims[1];
    int dim2 = input_tensor->dims[2];
    int dim3 = input_tensor->dims[3];
    int8_t* data = (int8_t*)input_tensor->data;
    int8_t* out_data = (int8_t*)output_tensor->data;
    fp16_t* slope_fp16 = (fp16_t*)slope_tensor->data;

    /* dequant */
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    float slope_scale = slope_tensor->scale;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;
    int slope_size = slope_tensor->elem_num;

    float* input_fp32 = (float*)sys_malloc(input_size * sizeof(float));
    float* output_fp32 = (float*)sys_malloc(output_size * sizeof(float));
    float* slope_fp32 = (float*)sys_malloc(slope_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input_fp32[i] = (float)data[i] * input_scale;
    }
    //目前int8模式下，Prelu的斜率使用的是float16类型（不是int8类型），所以进行计算前需要先将其从float16转为float32。
    for (int i = 0; i < slope_size; i++)
    {
        slope_fp32[i] = fp16_to_fp32(slope_fp16[i]);
    }
    int offset = 0;
    // nchw
    for (int i = 0; i < dim0; i++)
    {
        for (int c = 0; c < dim1; c++)
        {
            for (int l = 0; l < dim2; l++)
            {
                for (int k = 0; k < dim3; k++)
                {
                    // nchw
                    offset = i * dim1 * dim2 * dim3 + c * dim2 * dim3 + l * dim3 + k;
                    output_fp32[offset] = MAX(input_fp32[offset], 0) + slope_fp32[c] * MIN(input_fp32[offset], 0.f);
                }
            }
        }
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int data_i32 = round(output_fp32[i] / output_scale);
        if (data_i32 > 127)
            data_i32 = 127;
        else if (data_i32 < -127)
            data_i32 = -127;
        out_data[i] = (int8_t)data_i32;
    }

    sys_free(input_fp32);
    sys_free(output_fp32);
    sys_free(slope_fp32);

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

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    int ret = 0;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (input_tensor->dims[1] != output_tensor->dims[1] || input_tensor->dims[2] != output_tensor->dims[2] || input_tensor->dims[3] != output_tensor->dims[3])
        ret = set_ir_tensor_shape(output_tensor, input_tensor->dims, input_tensor->dim_num);

    return ret;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    struct tensor* slope_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    slope_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_prelu_fp32(input_tensor, output_tensor, slope_tensor);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_prelu_uint8(input_tensor, output_tensor, slope_tensor);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_prelu_int8(input_tensor, output_tensor, slope_tensor);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = reshape,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_prelu_ref_op()
{
    return register_builtin_node_ops(OP_PRELU, &hcl_node_ops);
}

int unregister_prelu_ref_op()
{
    return unregister_builtin_node_ops(OP_PRELU, &hcl_node_ops);
}
