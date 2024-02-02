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

#include "fc_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"
#include "utility/float.h"
#include "utility/sys_port.h"
#include "utility/log.h"

#include <math.h>
#include <string.h>

struct fc_data
{
    int need_trans;
    int batch;      // N
    int out_number; // OUT
    int hidden;     // hidden
    int zero[3];    // input, kernel, output
    float scale[3]; // input, kernel, output
};

static int ref_fc_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor, struct fc_data* param)
{
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

    float* input = (float*)input_tensor->data;
    float* output = (float*)output_tensor->data;
    float* weight = (float*)weight_tensor->data;
    float* bias = NULL;
    if (bias_tensor)
        bias = (float*)bias_tensor->data;

    int n, i, j;
    for (n = 0; n < batch; n++)
    {
        for (i = 0; i < out_number; i++)
        {
            float tmp = bias ? bias[i] : 0.f;
            for (j = 0; j < hidden; j++)
            {
                if (param->need_trans == 0)
                    tmp += input[n * hidden + j] * weight[i * hidden + j];
                else
                    tmp += input[n * hidden + j] * weight[i + j * out_number];
            }
            output[n * out_number + i] = tmp;
        }
    }

    return 0;
}

static int ref_fc_fp16(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor, struct fc_data* param)
{
#if MACOS

#else
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

    fp16_t* input = (fp16_t*)input_tensor->data;
    fp16_t* output = (fp16_t*)output_tensor->data;
    fp16_t* weight = (fp16_t*)weight_tensor->data;
    fp16_t* bias = NULL;
    if (bias_tensor)
        bias = (fp16_t*)bias_tensor->data;

    int n, i, j;
    for (n = 0; n < batch; n++)
    {
        for (i = 0; i < out_number; i++)
        {
            float tmp = bias ? fp16_to_fp32(bias[i]) : 0.f;
            for (j = 0; j < hidden; j++)
            {
                if (param->need_trans == 0)
                    tmp += fp16_to_fp32(input[n * hidden + j]) * fp16_to_fp32(weight[i * hidden + j]);
                else
                    tmp += fp16_to_fp32(input[n * hidden + j]) * fp16_to_fp32(weight[i + j * out_number]);
            }
            output[n * out_number + i] = fp32_to_fp16(tmp);
        }
    }
#endif
    return 0;
}

static int ref_fc_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor, struct fc_data* param)
{
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

    uint8_t* input = (uint8_t*)input_tensor->data;
    uint8_t* output = (uint8_t*)output_tensor->data;
    uint8_t* weight = (uint8_t*)weight_tensor->data;

    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    float weight_scale = weight_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t output_zero = output_tensor->zero_point;
    int32_t weight_zero = weight_tensor->zero_point;

    if (bias_tensor)
    {
        int32_t* bias = (int32_t*)bias_tensor->data;
        float bias_scale = bias_tensor->scale;

        int n, i, j;
        for (n = 0; n < batch; n++)
        {
            for (i = 0; i < out_number; i++)
            {
                float data = bias[i] * bias_scale;
                for (j = 0; j < hidden; j++)
                {
                    if (param->need_trans == 0)
                    {
                        float input_fp32 = ((float)input[n * hidden + j] - (float)input_zero) * input_scale;
                        float weight_fp32 = ((float)weight[i * hidden + j] - (float)weight_zero) * weight_scale;
                        data += input_fp32 * weight_fp32;
                    }
                    else
                    {
                        float input_fp32 = ((float)input[n * hidden + j] - (float)input_zero) * input_scale;
                        float weight_fp32 = ((float)weight[i + j * out_number] - (float)weight_zero) * weight_scale;
                        data += input_fp32 * weight_fp32;
                    }
                }
                int udata = roundf(data / output_scale) + output_zero;
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;
                output[n * out_number + i] = udata;
            }
        }
    }
    else
    {
        int n, i, j;
        for (n = 0; n < batch; n++)
        {
            for (i = 0; i < out_number; i++)
            {
                float data = 0.f;
                for (j = 0; j < hidden; j++)
                {
                    if (param->need_trans == 0)
                    {
                        float input_fp32 = ((float)input[n * hidden + j] - (float)input_zero) * input_scale;
                        float weight_fp32 = ((float)weight[i * hidden + j] - (float)weight_zero) * weight_scale;
                        data += input_fp32 * weight_fp32;
                    }
                    else
                    {
                        float input_fp32 = ((float)input[n * hidden + j] - (float)input_zero) * input_scale;
                        float weight_fp32 = ((float)weight[i + j * out_number] - (float)weight_zero) * weight_scale;
                        data += input_fp32 * weight_fp32;
                    }
                }
                int udata = round(data / output_scale) + output_zero;
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;
                output[n * out_number + i] = udata;
            }
        }
    }

    return 0;
}

static int ref_fc_int8(struct tensor* input_tensor, struct tensor* output_tensor, struct tensor* weight_tensor, struct tensor* bias_tensor, struct fc_data* param)
{
    int batch = param->batch;
    int hidden = param->hidden;
    int out_number = param->out_number;

    int8_t* input = (int8_t*)input_tensor->data;
    int8_t* output = (int8_t*)output_tensor->data;
    int8_t* weight = (int8_t*)weight_tensor->data;

    float input_scale = input_tensor->scale;
    float output_scale = output_tensor->scale;
    float* weight_scales = weight_tensor->scale_list;
    float* requant_scales = (float*)malloc(out_number * sizeof(float));

    for (int i = 0; i < out_number; i++)
        requant_scales[i] = (input_scale * weight_scales[i]) / output_scale;

    if (bias_tensor)
    {
        int32_t* bias_i32 = (int32_t*)bias_tensor->data;

        int n, i, j;
        for (n = 0; n < batch; n++)
        {
            for (i = 0; i < out_number; i++)
            {
                int32_t output_i32 = bias_i32[i];
                for (j = 0; j < hidden; j++)
                {
                    if (param->need_trans == 0)
                    {
                        int8_t input_i8 = input[n * hidden + j];
                        int8_t weight_i8 = weight[i * hidden + j];
                        output_i32 += (int32_t)input_i8 * (int32_t)weight_i8;
                    }
                    else
                    {
                        int8_t input_i8 = input[n * hidden + j];
                        int8_t weight_i8 = weight[i + j * out_number];
                        output_i32 += (int32_t)input_i8 * (int32_t)weight_i8;
                    }
                }
                int data_i32 = roundf(output_i32 * requant_scales[i]);
                if (data_i32 > 127)
                    data_i32 = 127;
                else if (data_i32 < -127)
                    data_i32 = -127;
                output[n * out_number + i] = (int8_t)data_i32;
            }
        }
    }
    else
    {
        int n, i, j;
        for (n = 0; n < batch; n++)
        {
            for (i = 0; i < out_number; i++)
            {
                int32_t output_i32 = 0;
                for (j = 0; j < hidden; j++)
                {
                    if (param->need_trans == 0)
                    {
                        int8_t input_i8 = input[n * hidden + j];
                        int8_t weight_i8 = weight[i * hidden + j];
                        output_i32 += (int32_t)input_i8 * (int32_t)weight_i8;
                    }
                    else
                    {
                        int8_t input_i8 = input[n * hidden + j];
                        int8_t weight_i8 = weight[i + j * out_number];
                        output_i32 += (int32_t)input_i8 * (int32_t)weight_i8;
                    }
                }
                int data_i32 = roundf(output_i32 * requant_scales[i]);
                if (data_i32 > 127)
                    data_i32 = 127;
                else if (data_i32 < -127)
                    data_i32 = -127;
                output[n * out_number + i] = (int8_t)data_i32;
            }
        }
    }

    free(requant_scales);

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct fc_data* op_param = (struct fc_data*)sys_malloc(sizeof(struct fc_data));
    memset(op_param, 0, sizeof(struct fc_data));
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
    struct tensor* weight_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_param* param = (struct fc_param*)ir_node->op.param_mem;
    struct fc_data* op_param = (struct fc_data*)exec_node->ops_priv;

    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        int hidden = input_tensor->dims[1];
        if (input_tensor->dim_num > 2)
            hidden = hidden * input_tensor->dims[2];
        if (input_tensor->dim_num > 3)
            hidden = hidden * input_tensor->dims[3];
        op_param->hidden = hidden;
    }
    else
    {
        int hidden = 0;
        if (input_tensor->dim_num == 2)
            hidden = input_tensor->dims[1];
        if (input_tensor->dim_num == 3)
            hidden = input_tensor->dims[1] * input_tensor->dims[2];
        if (input_tensor->dim_num == 4)
            hidden = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];
        op_param->hidden = hidden;
    }
    op_param->batch = input_tensor->dims[0];
    op_param->out_number = param->num_output;

    int weight_out = weight_tensor->dims[0];

    if (weight_out == op_param->out_number)
        op_param->need_trans = 0;
    else
        op_param->need_trans = 1;

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* bias_tensor = NULL;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_param* param = (struct fc_param*)ir_node->op.param_mem;
    struct fc_data* op_param = (struct fc_data*)exec_node->ops_priv;

    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_fc_fp32(input_tensor, output_tensor, weight_tensor, bias_tensor, op_param);
    else if (input_tensor->data_type == TENGINE_DT_FP16)
#if MACOS
        TLOG_ERR("FP16 not support for mac os");
#else
        ret = ref_fc_fp16(input_tensor, output_tensor, weight_tensor, bias_tensor, op_param);
#endif
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_fc_uint8(input_tensor, output_tensor, weight_tensor, bias_tensor, op_param);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_fc_int8(input_tensor, output_tensor, weight_tensor, bias_tensor, op_param);
    else
    {
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);
        return -1;
    }

    return ret;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int dim[4];

    int n = weight->dims[0];
    int k = weight->dims[1];

    int m = input->dims[0];
    int input_k = input->dims[1];

    if (input->dim_num == 2)
    {
        dim[0] = m;
        dim[1] = n;
    }
    else if (input->dim_num == 3)
    {
        if (input->dims[2] != 0)
            input_k *= input->dims[2];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
        }
    }
    else if (input->dim_num == 4)
    {
        if (input->dims[2] * input->dims[3] != 0)
            input_k *= input->dims[2] * input->dims[3];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = 1;
            dim[3] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
            dim[3] = 1;
        }
    }
    else
        return -1;

    if (k != input_k)
    {
        TLOG_ERR("fc: input tensor and weight tensor shape does not match, hidden_number: %d\n", k);
        return -1;
    }

    int ret = set_ir_tensor_shape(output, dim, input->dim_num);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_fc_ref_op()
{
    return register_builtin_node_ops(OP_FC, &hcl_node_ops);
}

int unregister_fc_ref_op()
{
    return unregister_builtin_node_ops(OP_FC, &hcl_node_ops);
}
