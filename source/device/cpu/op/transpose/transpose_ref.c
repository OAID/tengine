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

#include "transpose_param.h"

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

struct ref_transpose_param
{
    int* in_dims;
    int* permute;
    int dims;
};

void transpose2d(float* input, float* output, const struct ref_transpose_param* param)
{
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]];
    int in_dim1 = param->in_dims[1];
    int inStride[2];
    inStride[0] = in_dim1;
    inStride[1] = 1;

    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];

    for (int n = 0; n < out_dim0; n++)
    { // 1
        for (int h = 0; h < out_dim1; h++)
        { // 1
            output[n * out_dim1 + h] = input[n * stride0 + h * stride1];
        }
    }
}

void transpose3d(float* input, float* output, const struct ref_transpose_param* param)
{
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]];
    int out_dim2 = param->in_dims[param->permute[2]];

    int in_dim1 = param->in_dims[1];
    int in_dim2 = param->in_dims[2];

    int inStride[3];
    inStride[0] = in_dim1 * in_dim2;
    inStride[1] = in_dim2;
    inStride[2] = 1;

    int outStride0 = out_dim1 * out_dim2;
    int outStride1 = out_dim2;

    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];
    int stride2 = inStride[param->permute[2]];

    for (int n = 0; n < out_dim0; n++)
    { // 1
        for (int h = 0; h < out_dim1; h++)
        { // 1
            for (int w = 0; w < out_dim2; w++)
            { // 2
                output[n * outStride0 + h * outStride1 + w] = input[n * stride0 + h * stride1 + w * stride2];
            }
        }
    }
    return;
}

void transpose4d(float* input, float* output, const struct ref_transpose_param* param)
{
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]];
    int out_dim2 = param->in_dims[param->permute[2]];
    int out_dim3 = param->in_dims[param->permute[3]];

    int in_dim1 = param->in_dims[1];
    int in_dim2 = param->in_dims[2];
    int in_dim3 = param->in_dims[3];

    int inStride[4];
    inStride[0] = in_dim1 * in_dim2 * in_dim3;
    inStride[1] = in_dim2 * in_dim3;
    inStride[2] = in_dim3;
    inStride[3] = 1;

    int outStride0 = out_dim1 * out_dim2 * out_dim3;
    int outStride1 = out_dim2 * out_dim3;
    int outStride2 = out_dim3;

    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];
    int stride2 = inStride[param->permute[2]];
    int stride3 = inStride[param->permute[3]];

    for (int n = 0; n < out_dim0; n++)
    { // 1
        for (int h = 0; h < out_dim1; h++)
        { // 1
            for (int w = 0; w < out_dim2; w++)
            { // 2
                for (int c = 0; c < out_dim3; c++)
                { // 2
                    output[n * outStride0 + h * outStride1 + w * outStride2 + c] = input[n * stride0 + h * stride1 + w * stride2 + c * stride3];
                }
            }
        }
    }
}
void transpose5d(float* input, float* output, const struct ref_transpose_param* param)
{
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]];
    int out_dim2 = param->in_dims[param->permute[2]];
    int out_dim3 = param->in_dims[param->permute[3]];
    int out_dim4 = param->in_dims[param->permute[4]];

    int in_dim1 = param->in_dims[1];
    int in_dim2 = param->in_dims[2];
    int in_dim3 = param->in_dims[3];
    int in_dim4 = param->in_dims[4];

    int inStride[5];
    inStride[0] = in_dim1 * in_dim2 * in_dim3 * in_dim4;
    inStride[1] = in_dim2 * in_dim3 * in_dim4;
    inStride[2] = in_dim3 * in_dim4;
    inStride[3] = in_dim4;
    inStride[4] = 1;

    int outStride0 = out_dim1 * out_dim2 * out_dim3 * out_dim4;
    int outStride1 = out_dim2 * out_dim3 * out_dim4;
    int outStride2 = out_dim3 * out_dim4;
    int outStride3 = out_dim4;

    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];
    int stride2 = inStride[param->permute[2]];
    int stride3 = inStride[param->permute[3]];
    int stride4 = inStride[param->permute[4]];

    for (int n = 0; n < out_dim0; n++)
    { // 1
        for (int h = 0; h < out_dim1; h++)
        { // 1
            for (int w = 0; w < out_dim2; w++)
            { // 2
                for (int c = 0; c < out_dim3; c++)
                { // 2
                    for (int x = 0; x < out_dim4; x++)
                    {
                        output[n * outStride0 + h * outStride1 + w * outStride2 + c * outStride3 + x] = input[n * stride0 + h * stride1 + w * stride2 + c * stride3 + x * stride4];
                    }
                }
            }
        }
    }
}

void transpose6d(float* input, float* output, const struct ref_transpose_param* param)
{
    int out_dim0 = param->in_dims[param->permute[0]];
    int out_dim1 = param->in_dims[param->permute[1]];
    int out_dim2 = param->in_dims[param->permute[2]];
    int out_dim3 = param->in_dims[param->permute[3]];
    int out_dim4 = param->in_dims[param->permute[4]];
    int out_dim5 = param->in_dims[param->permute[5]];

    int in_dim1 = param->in_dims[1];
    int in_dim2 = param->in_dims[2];
    int in_dim3 = param->in_dims[3];
    int in_dim4 = param->in_dims[4];
    int in_dim5 = param->in_dims[5];

    int inStride[6];
    inStride[0] = in_dim1 * in_dim2 * in_dim3 * in_dim4 * in_dim5;
    inStride[1] = in_dim2 * in_dim3 * in_dim4 * in_dim5;
    inStride[2] = in_dim3 * in_dim4 * in_dim5;
    inStride[3] = in_dim4 * in_dim5;
    inStride[4] = in_dim5;
    inStride[5] = 1;

    int outStride0 = out_dim1 * out_dim2 * out_dim3 * out_dim4 * out_dim5;
    int outStride1 = out_dim2 * out_dim3 * out_dim4 * out_dim5;
    int outStride2 = out_dim3 * out_dim4 * out_dim5;
    int outStride3 = out_dim4 * out_dim5;
    int outStride4 = out_dim5;

    int stride0 = inStride[param->permute[0]];
    int stride1 = inStride[param->permute[1]];
    int stride2 = inStride[param->permute[2]];
    int stride3 = inStride[param->permute[3]];
    int stride4 = inStride[param->permute[4]];
    int stride5 = inStride[param->permute[5]];

    for (int n = 0; n < out_dim0; n++)
    { // 1
        for (int h = 0; h < out_dim1; h++)
        { // 1
            for (int w = 0; w < out_dim2; w++)
            { // 2
                for (int c = 0; c < out_dim3; c++)
                { // 2
                    for (int x = 0; x < out_dim4; x++)
                    {
                        for (int y = 0; y < out_dim5; y++)
                        {
                            output[n * outStride0 + h * outStride1 + w * outStride2 + c * outStride3 + x * outStride4 + y] = input[n * stride0 + h * stride1 + w * stride2 + c * stride3 + x * stride4 + y * stride5];
                        }
                    }
                }
            }
        }
    }
}

static int ref_transpose_fp32(float* input, float* output, const struct ref_transpose_param* param)
{
    switch (param->dims)
    {
    case 2:
        transpose2d(input, output, param);
        break;
    case 3:
        transpose3d(input, output, param);
        break;
    case 4:
        transpose4d(input, output, param);
        break;
    case 5:
        transpose5d(input, output, param);
        break;
    case 6:
        transpose6d(input, output, param);
        break;
    default:
        break;
    }
    return 0;
}

static int ref_transpose_int8(struct tensor* input_tensor, struct tensor* output_tensor, const struct ref_transpose_param* param)
{
    /* dequant */
    int8_t* input_int8 = (int8_t*)input_tensor->data;
    int8_t* output_int8 = (int8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int input_size = input_tensor->elem_num;
    int output_size = output_tensor->elem_num;

    float* input = (float*)sys_malloc(input_size * sizeof(float));
    float* output = (float*)sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input[i] = ((float)input_int8[i] - (float)input_zero) * input_scale;
    }

    switch (param->dims)
    {
    case 2:
        transpose2d(input, output, param);
        break;
    case 3:
        transpose3d(input, output, param);
        break;
    case 4:
        transpose4d(input, output, param);
        break;
    case 5:
        transpose5d(input, output, param);
        break;
    case 6:
        transpose6d(input, output, param);
        break;
    default:
        break;
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int idata = round(output[i] / output_scale + output_zero);
        if (idata > 127)
            idata = 127;
        else if (idata < -127)
            idata = -127;
        output_int8[i] = idata;
    }

    sys_free(input);
    sys_free(output);

    return 0;
}

static int ref_transpose_uint8(struct tensor* input_tensor, struct tensor* output_tensor, const struct ref_transpose_param* param)
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

    float* input = (float*)sys_malloc(input_size * sizeof(float));
    float* output = (float*)sys_malloc(output_size * sizeof(float));

    for (int i = 0; i < input_size; i++)
    {
        input[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    switch (param->dims)
    {
    case 2:
        transpose2d(input, output, param);
        break;
    case 3:
        transpose3d(input, output, param);
        break;
    case 4:
        transpose4d(input, output, param);
        break;
    case 5:
        transpose5d(input, output, param);
        break;
    case 6:
        transpose6d(input, output, param);
        break;
    default:
        break;
    }

    /* quant */
    for (int i = 0; i < output_size; i++)
    {
        int udata = round(output[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    sys_free(input);
    sys_free(output);

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ref_transpose_param* op_param = (struct ref_transpose_param*)sys_malloc(sizeof(struct ref_transpose_param));
    memset(op_param, 0, sizeof(struct ref_transpose_param));
    exec_node->ops_priv = op_param;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct ref_transpose_param* op_param = (struct ref_transpose_param*)exec_node->ops_priv;
    struct transpose_param* transpose_param = (struct transpose_param*)ir_node->op.param_mem;
    int tr_size = transpose_param->tr_shape_size;
    // int tr_size = 2 ;
    op_param->permute = (int*)sys_malloc(tr_size * sizeof(int));
    op_param->dims = input_tensor->dim_num;
    op_param->in_dims = (int*)sys_malloc(op_param->dims * sizeof(int));

    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct ref_transpose_param* op_param = (struct ref_transpose_param*)exec_node->ops_priv;

    sys_free(op_param->permute);
    sys_free(op_param->in_dims);

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
    struct transpose_param* transpose_param = (struct transpose_param*)ir_node->op.param_mem;

    void* out_data = (void*)output_tensor->data;
    void* in_data = (void*)input_tensor->data;

    struct ref_transpose_param* op_param = (struct ref_transpose_param*)exec_node->ops_priv;

    int tr_size = transpose_param->tr_shape_size;

    for (int i = 0; i < tr_size; i++)
    {
        op_param->permute[i] = transpose_param->tr_shape[i];
    }

    for (int i = 0; i < (int)op_param->dims; i++)
    {
        op_param->in_dims[i] = input_tensor->dims[i];
    }

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_transpose_fp32((float*)in_data, (float*)out_data, op_param);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_transpose_int8(input_tensor, output_tensor, op_param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_transpose_uint8(input_tensor, output_tensor, op_param);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_transpose_ref_op()
{
    return register_builtin_node_ops(OP_TRANSPOSE, &hcl_node_ops);
}

int unregister_transpose_ref_op()
{
    return unregister_builtin_node_ops(OP_TRANSPOSE, &hcl_node_ops);
}
