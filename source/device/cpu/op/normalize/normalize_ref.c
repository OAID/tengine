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
 * Author: jxyang@openailab.com
 */

#include "normalize_param.h"

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

static void norm_channel(float* input, float* output, float* buffer, float* scale, int hw, int channel, int num_thread)
{
    memset(buffer, 0, hw * sizeof(float));

    //#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < channel; i++)
    {
        for (int j = 0; j < hw; j++)
        {
            float data = *(input + i * hw + j);
            buffer[j] += (data * data);
        }
    }

    //#pragma omp parallel for num_threads(num_thread)
    for (int j = 0; j < hw; j++)
    {
        buffer[j] = 1.f / sqrt(buffer[j]);
    }

    //#pragma omp parallel for num_threads(num_thread)
    for (int i = 0; i < channel; i++)
    {
        for (int j = 0; j < hw; j++)
        {
            float data = *(input + i * hw + j);
            *(output + i * hw + j) = data * buffer[j] * scale[i];
        }
    }
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
    struct tensor* scale_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    normalize_param_t* param = (normalize_param_t*)(ir_node->op.param_mem);
    float* input_org = (float*)input_tensor->data;
    float* output_org = (float*)output_tensor->data;
    float* sclae_org = (float*)scale_tensor->data;

    int batch_number = input_tensor->dims[0];
    int channel_num = input_tensor->dims[1];
    int channel_size = (input_tensor->dims[2]) * (input_tensor->dims[3]);
    int img_size = channel_num * channel_size;

    float* buffer = (float*)sys_malloc(channel_size * sizeof(float));
    if (param->channel_shared == 0 && param->across_spatial == 0)
    {
        for (int i = 0; i < batch_number; i++)
        {
            norm_channel(input_org, output_org, buffer, sclae_org, channel_size, channel_num, exec_graph->num_thread);
            input_org += img_size;
            output_org += img_size;
        }
    }

    sys_free(buffer);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops normalize_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_normalize_ref_op()
{
    return register_builtin_node_ops(OP_NORMALIZE, &normalize_node_ops);
}

int unregister_normalize_ref_op()
{
    return unregister_builtin_node_ops(OP_NORMALIZE, &normalize_node_ops);
}
