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

#include "shuffle_channel_param.h"

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

int ref_shuffle_channel_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct shuffle_channel_param* param)
{
    int batch = input_tensor->dims[0];
    int c = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int group = param->group;
    int elemsize = input_tensor->elem_size;
    int chs_per_group = c / group;

    float* input_fp32 = (float*)input_tensor->data;
    float* output_fp32 = (float*)output_tensor->data;

    for (int n = 0; n < batch; n++)
    {
        for (int i = 0; i < group; i++)
        {
            for (int j = 0; j != chs_per_group; j++)
            {
                int src_q = n * c * h * w + (chs_per_group * i + j) * h * w;
                int dst_q = n * c * h * w + (group * j + i) * h * w;
                memcpy(output_fp32 + dst_q, input_fp32 + src_q, (size_t)h * w * elemsize);
            }
        }
    }

    return 0;
}

int ref_shuffle_channel_uint8(struct tensor* input_tensor, struct tensor* output_tensor, struct shuffle_channel_param* param)
{
    int batch = input_tensor->dims[0];
    int c = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int group = param->group;
    int elemsize = input_tensor->elem_size;
    int chs_per_group = c / group;

    uint8_t* input_uint8 = (uint8_t*)input_tensor->data;
    uint8_t* output_uint8 = (uint8_t*)output_tensor->data;

    for (int n = 0; n < batch; n++)
    {
        for (int i = 0; i < group; i++)
        {
            for (int j = 0; j != chs_per_group; j++)
            {
                int src_q = n * c * h * w + (chs_per_group * i + j) * h * w;
                int dst_q = n * c * h * w + (group * j + i) * h * w;
                memcpy(output_uint8 + dst_q, input_uint8 + src_q, (size_t)h * w * elemsize);
            }
        }
    }

    return 0;
}

int ref_shuffle_channel_int8(struct tensor* input_tensor, struct tensor* output_tensor, struct shuffle_channel_param* param)
{
    int batch = input_tensor->dims[0];
    int c = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];
    int group = param->group;
    int elemsize = input_tensor->elem_size;
    int chs_per_group = c / group;

    int8_t* input_int8 = (int8_t*)input_tensor->data;
    int8_t* output_int8 = (int8_t*)output_tensor->data;

    for (int n = 0; n < batch; n++)
    {
        for (int i = 0; i < group; i++)
        {
            for (int j = 0; j != chs_per_group; j++)
            {
                int src_q = n * c * h * w + (chs_per_group * i + j) * h * w;
                int dst_q = n * c * h * w + (group * j + i) * h * w;
                memcpy(output_int8 + dst_q, input_int8 + src_q, (size_t)h * w * elemsize);
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

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    if (input_tensor->dim_num != 4)
    {
        TLOG_ERR("dims num is not 4, not support shuffle channel\n");
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct shuffle_channel_param* param = (struct shuffle_channel_param*)ir_node->op.param_mem;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_shuffle_channel_fp32(input_tensor, output_tensor, param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_shuffle_channel_uint8(input_tensor, output_tensor, param);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_shuffle_channel_int8(input_tensor, output_tensor, param);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
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

int register_shuffle_channel_ref_op()
{
    return register_builtin_node_ops(OP_SHUFFLECHANNEL, &hcl_node_ops);
}

int unregister_shuffle_channel_ref_op()
{
    return unregister_builtin_node_ops(OP_SHUFFLECHANNEL, &hcl_node_ops);
}
