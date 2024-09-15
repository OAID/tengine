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

#include "spatialtransformer_param.h"

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

int between(float value, float lowerBound, float upperBound)
{
    if (value >= lowerBound && value <= upperBound)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

int BilinearSampling(int o_n, int o_c, int o_h, int o_w, int i_c, int i_h, int i_w, float* in_data, float* out_data, float* grid_total)
{
    float* tmp_out = out_data;
    for (int n = 0; n < o_n; n++)
    {
        for (int c = 0; c < o_c; c++)
        {
            for (int h = 0; h < o_h; h++)
            {
                for (int w = 0; w < o_w; w++)
                {
                    int out_index = n * o_c * o_h * o_w + c * o_h * o_w + h * o_w + w;
                    int grid_index = n * o_h * o_w * 2 + h * o_w + w;
                    float y_real = (*(grid_total + grid_index + o_h * o_w) + 1.0) * (i_h - 1.0) / 2.0;
                    float x_real = (*(grid_total + grid_index) + 1.0) * (i_w - 1.0) / 2.0;
                    int top_left_y = floor(y_real);
                    int top_left_x = floor(x_real);
                    float top_left_y_w = 1.0 - (y_real - top_left_y);
                    float top_left_x_w = 1.0 - (x_real - top_left_x);
                    int data_index = n * i_c * i_h * i_w + c * i_h * i_w + top_left_y * i_w + top_left_x;
                    float top_left_v = 0;
                    float top_right_v = 0;
                    float bottom_left_v = 0;
                    float bottom_right_v = 0;
                    int lower_bound = 0;
                    if (between(top_left_x, lower_bound, i_w - 1) && between(top_left_y, lower_bound, i_h - 1))
                    {
                        top_left_v = *(in_data + data_index);
                    }
                    if (between(top_left_x + 1, lower_bound, i_w - 1) && between(top_left_y, lower_bound, i_h - 1))
                    {
                        top_right_v = *(in_data + data_index + 1);
                    }
                    if (between(top_left_x, lower_bound, i_w - 1) && between(top_left_y + 1, lower_bound, i_h - 1))
                    {
                        bottom_left_v = *(in_data + data_index + i_w);
                    }
                    if (between(top_left_x + 1, lower_bound, i_w - 1) && between(top_left_y + 1, lower_bound, i_h - 1))
                    {
                        bottom_right_v = *(in_data + data_index + i_w + 1);
                    }
                    *(tmp_out + out_index) = top_left_v * top_left_y_w * top_left_x_w + top_right_v * top_left_y_w * (1.0 - top_left_x_w) + bottom_left_v * (1.0 - top_left_y_w) * top_left_x_w + bottom_right_v * (1.0 - top_left_y_w) * (1.0 - top_left_x_w);
                }
            }
        }
    }
    return 1;
}

int ref_spatialtransformer_uint8(struct tensor* input_tensor, struct tensor* input_tensor1, struct tensor* output_tensor,
                                 struct spatialtransformer_param* param, int num_thread)
{
    int indices_dim_size = input_tensor->dim_num;

    int total_size = input_tensor->elem_num;
    int loc_size = input_tensor1->elem_num;

    /* dequant */
    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* loc_uint8 = (uint8_t*)input_tensor1->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;
    float input_scale = input_tensor->scale;
    float loc_scale = input_tensor1->scale;
    float output_scale = output_tensor->scale;
    int32_t input_zero = input_tensor->zero_point;
    int32_t loc_zero = input_tensor1->zero_point;
    int32_t output_zero = output_tensor->zero_point;

    float* in_data = (float*)malloc(total_size * sizeof(float));
    float* out_data = (float*)malloc(sizeof(float) * 3 * param->target_shape[0] * param->target_shape[1]);
    float* loc_data = (float*)malloc(loc_size * sizeof(float));

    for (int i = 0; i < total_size; i++)
    {
        in_data[i] = ((float)input_uint8[i] - (float)input_zero) * input_scale;
    }

    for (int i = 0; i < loc_size; i++)
    {
        loc_data[i] = ((float)loc_uint8[i] - (float)loc_zero) * loc_scale;
    }

    int batch = input_tensor->dims[1];
    float* workspace = (float*)malloc(sizeof(float) * 3 * param->target_shape[0] * param->target_shape[1]);
    int target_shape_hw = param->target_shape[0] * param->target_shape[1];
    for (int i = 1; i <= target_shape_hw; i++)
    {
        workspace[0 * target_shape_hw + i - 1] = -1.0 + (i - 1) % param->target_shape[1] * 2.0 / (param->target_shape[1] - 1);
        workspace[1 * target_shape_hw + i - 1] = -1.0 + (i - 1) / param->target_shape[1] * 2.0 / (param->target_shape[0] - 1);
        workspace[2 * target_shape_hw + i - 1] = 1.0;
    }
    int m = 2;
    int p = target_shape_hw;
    int n = 3;

    float* grid_src = (float*)malloc(sizeof(float) * 2 * target_shape_hw * batch);
    float* grid_dst = (float*)malloc(sizeof(float) * 3 * target_shape_hw);

    for (int i = 0; i < 3 * target_shape_hw; i++)
    {
        grid_dst[i] = workspace[i];
    }
    if (param->transformer_type == 0)
    { // Affine
        for (int b = 0; b < batch; b++)
        {
            int index = b * target_shape_hw;
            float* grid_src_batch = grid_src + 0;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < target_shape_hw; j++)
                {
                    grid_src_batch[i * p + j] = 0;
                    for (int a = 1; a <= n; a++)
                    {
                        grid_src_batch[i * p + j] += loc_data[i * n + a - 1] * grid_dst[(a - 1) * p + j];
                    }
                }
            }
        }
    }

    if (param->sampler_type == 1)
    { // Bilinear
        int o_n = output_tensor->dims[0];
        int o_c = output_tensor->dims[1];
        int o_h = output_tensor->dims[2];
        int o_w = output_tensor->dims[3];
        int i_c = input_tensor->dims[1];
        int i_h = input_tensor->dims[2];
        int i_w = input_tensor->dims[3];
        int ret = BilinearSampling(o_n, o_c, o_h, o_w, i_c, i_h, i_w, in_data, out_data, grid_src);
    }
    else
    {
        TLOG_ERR("Extra type not support yet\n");
    }

    /* quant */
    for (int i = 0; i < total_size; i++)
    {
        int udata = round(out_data[i] / output_scale + output_zero);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        output_uint8[i] = udata;
    }

    free(in_data);
    free(out_data);
    free(loc_data);

    free(grid_src);
    free(grid_dst);
    free(workspace);
    return 0;
}

int ref_spatialtransformer_fp32(struct tensor* input_tensor, struct tensor* input_tensor1, struct tensor* output_tensor,
                                struct spatialtransformer_param* param, int num_thread)
{
    int indices_dim_size = input_tensor->dim_num;

    float* in_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;
    float* loc_data = (float*)input_tensor1->data;

    int batch = input_tensor->dims[1];

    float* workspace = (float*)malloc(sizeof(float) * 3 * param->target_shape[0] * param->target_shape[1]);

    int target_shape_hw = param->target_shape[0] * param->target_shape[1];
    for (int i = 1; i <= target_shape_hw; i++)
    {
        workspace[0 * target_shape_hw + i - 1] = -1.0 + (i - 1) % param->target_shape[1] * 2.0 / (param->target_shape[1] - 1);
        workspace[1 * target_shape_hw + i - 1] = -1.0 + (i - 1) / param->target_shape[1] * 2.0 / (param->target_shape[0] - 1);
        workspace[2 * target_shape_hw + i - 1] = 1.0;
    }
    int m = 2;
    int p = target_shape_hw;
    int n = 3;

    float* grid_src = (float*)malloc(sizeof(float) * 2 * target_shape_hw * batch);
    float* grid_dst = (float*)malloc(sizeof(float) * 3 * target_shape_hw);

    for (int i = 0; i < 3 * target_shape_hw; i++)
    {
        grid_dst[i] = workspace[i];
    }
    if (param->transformer_type == 0)
    { // Affine
        for (int b = 0; b < batch; b++)
        {
            int index = b * target_shape_hw;
            float* grid_src_batch = grid_src + 0;
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < target_shape_hw; j++)
                {
                    grid_src_batch[i * p + j] = 0;
                    for (int a = 1; a <= n; a++)
                    {
                        grid_src_batch[i * p + j] += loc_data[i * n + a - 1] * grid_dst[(a - 1) * p + j];
                    }
                }
            }
        }
    }

    if (param->sampler_type == 1)
    { // Bilinear
        int o_n = output_tensor->dims[0];
        int o_c = output_tensor->dims[1];
        int o_h = output_tensor->dims[2];
        int o_w = output_tensor->dims[3];
        int i_c = input_tensor->dims[1];
        int i_h = input_tensor->dims[2];
        int i_w = input_tensor->dims[3];
        int ret = BilinearSampling(o_n, o_c, o_h, o_w, i_c, i_h, i_w, in_data, out_data, grid_src);
    }
    else
    {
        TLOG_ERR("Extra type not support yet\n");
    }

    free(grid_src);
    free(grid_dst);
    free(workspace);
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
    struct tensor* input_tensor1;
    struct tensor* output_tensor;
    int layout = ir_graph->graph_layout;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    int indices_dim_size = input_tensor1->dim_num;

    struct spatialtransformer_param* spatialtransformer_param = (struct spatialtransformer_param*)ir_node->op.param_mem;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_spatialtransformer_uint8(input_tensor, input_tensor1, output_tensor,
                                           spatialtransformer_param, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_spatialtransformer_fp32(input_tensor, input_tensor1, output_tensor,
                                          spatialtransformer_param, exec_graph->num_thread);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);
    if (ret != 0)
        return -1;

    return 0;
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

int register_spatialtransformer_ref_op()
{
    return register_builtin_node_ops(OP_SPATIALTRANSFORMER, &hcl_node_ops);
}

int unregister_spatialtransformer_ref_op()
{
    return unregister_builtin_node_ops(OP_SPATIALTRANSFORMER, &hcl_node_ops);
}
