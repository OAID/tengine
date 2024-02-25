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

#include "crop_param.h"

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

static int ref_crop_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct crop_param* param,
                         int num_thread)
{
    float* input = (float*)input_tensor->data;
    float* output = (float*)output_tensor->data;

    int iDataC = input_tensor->dims[1];
    int iDataH = input_tensor->dims[2];
    int iDataW = input_tensor->dims[3];

    int oDataN = output_tensor->dims[0];
    int oDataC = output_tensor->dims[1];
    int oDataH = output_tensor->dims[2];
    int oDataW = output_tensor->dims[3];

    // MXNet
    if (param->flag == 1)
    {
        if (param->num_args == 1)
        {
            int offsetH = (iDataH - param->crop_h) / 2;
            int offsetW = (iDataW - param->crop_w) / 2;
            if ((param->offset_h + oDataH <= iDataH) && (param->offset_w + oDataW <= iDataW))
            {
                for (int n = 0; n < oDataN; n++)
                {
                    for (int c = 0; c < oDataC; c++)
                    {
                        for (int h = 0; h < oDataH; h++)
                        {
                            int i_h = h + offsetH;
                            for (int w = 0; w < oDataW; w++)
                            {
                                int i_w = w + offsetW;
                                output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] = input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                            }
                        }
                    }
                }
            }
        }
        if (param->num_args == 2)
        {
            if ((param->offset_h + oDataH <= iDataH) && (param->offset_w + oDataW <= iDataW))
            {
                for (int n = 0; n < oDataN; n++)
                {
                    for (int c = 0; c < oDataC; c++)
                    {
                        for (int h = 0; h < oDataH; h++)
                        {
                            int i_h = h + param->offset_h;
                            for (int w = 0; w < oDataW; w++)
                            {
                                int i_w = w + param->offset_w;
                                output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] = input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                            }
                        }
                    }
                }
            }
        }
    }
    // Caffe
    if (param->flag == 0)
    {
        if (param->axis == 1)
        {
            for (int n = 0; n < oDataN; n++)
            {
                for (int c = 0; c < oDataC; c++)
                {
                    int i_c = param->offset_c + c;
                    for (int h = 0; h < oDataH; h++)
                    {
                        int i_h = param->offset_h + h;
                        for (int w = 0; w < oDataW; w++)
                        {
                            int i_w = param->offset_w + w;
                            output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] = input[n * iDataC * iDataH * iDataW + i_c * iDataH * iDataW + i_h * iDataW + i_w];
                        }
                    }
                }
            }
        }
        if (param->axis == 2)
        {
            for (int n = 0; n < oDataN; n++)
            {
                for (int c = 0; c < oDataC; c++)
                {
                    for (int h = 0; h < oDataH; h++)
                    {
                        int i_h = param->offset_h + h;
                        for (int w = 0; w < oDataW; w++)
                        {
                            int i_w = param->offset_w + w;
                            output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] = input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                        }
                    }
                }
            }
        }
    }

    return 0;
}

static int ref_crop_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct crop_param* param,
                          int num_thread)
{
    uint8_t* input = (uint8_t*)input_tensor->data;
    uint8_t* output = (uint8_t*)output_tensor->data;

    int iDataC = input_tensor->dims[1];
    int iDataH = input_tensor->dims[2];
    int iDataW = input_tensor->dims[3];

    int oDataN = output_tensor->dims[0];
    int oDataC = output_tensor->dims[1];
    int oDataH = output_tensor->dims[2];
    int oDataW = output_tensor->dims[3];

    // MXNet
    if (param->flag == 1)
    {
        if (param->num_args == 1)
        {
            int offsetH = (iDataH - param->crop_h) / 2;
            int offsetW = (iDataW - param->crop_w) / 2;
            if ((param->offset_h + oDataH <= iDataH) && (param->offset_w + oDataW <= iDataW))
            {
                for (int n = 0; n < oDataN; n++)
                {
                    for (int c = 0; c < oDataC; c++)
                    {
                        for (int h = 0; h < oDataH; h++)
                        {
                            int i_h = h + offsetH;
                            for (int w = 0; w < oDataW; w++)
                            {
                                int i_w = w + offsetW;
                                output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] = input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                            }
                        }
                    }
                }
            }
        }
        if (param->num_args == 2)
        {
            if ((param->offset_h + oDataH <= iDataH) && (param->offset_w + oDataW <= iDataW))
            {
                for (int n = 0; n < oDataN; n++)
                {
                    for (int c = 0; c < oDataC; c++)
                    {
                        for (int h = 0; h < oDataH; h++)
                        {
                            int i_h = h + param->offset_h;
                            for (int w = 0; w < oDataW; w++)
                            {
                                int i_w = w + param->offset_w;
                                output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] = input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                            }
                        }
                    }
                }
            }
        }
    }
    // Caffe
    if (param->flag == 0)
    {
        if (param->axis == 1)
        {
            for (int n = 0; n < oDataN; n++)
            {
                for (int c = 0; c < oDataC; c++)
                {
                    int i_c = param->offset_c + c;
                    for (int h = 0; h < oDataH; h++)
                    {
                        int i_h = param->offset_h + h;
                        for (int w = 0; w < oDataW; w++)
                        {
                            int i_w = param->offset_w + w;
                            output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] = input[n * iDataC * iDataH * iDataW + i_c * iDataH * iDataW + i_h * iDataW + i_w];
                        }
                    }
                }
            }
        }
        if (param->axis == 2)
        {
            for (int n = 0; n < oDataN; n++)
            {
                for (int c = 0; c < oDataC; c++)
                {
                    for (int h = 0; h < oDataH; h++)
                    {
                        int i_h = param->offset_h + h;
                        for (int w = 0; w < oDataW; w++)
                        {
                            int i_w = param->offset_w + w;
                            output[n * oDataC * oDataH * oDataW + c * oDataH * oDataW + h * oDataW + w] = input[n * iDataC * iDataH * iDataW + c * iDataH * iDataW + i_h * iDataW + i_w];
                        }
                    }
                }
            }
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
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct crop_param* crop_param = (struct crop_param*)ir_node->op.param_mem;

    if (input_tensor->data_type == TENGINE_DT_FP32)
        ref_crop_fp32(input_tensor, output_tensor, crop_param, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ref_crop_uint8(input_tensor, output_tensor, crop_param, exec_graph->num_thread);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_crop_ref_op()
{
    return register_builtin_node_ops(OP_CROP, &hcl_node_ops);
}

int unregister_crop_ref_op()
{
    return unregister_builtin_node_ops(OP_CROP, &hcl_node_ops);
}
