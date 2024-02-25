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

#include "region_param.h"

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

static int entry_index(int batch, int location, int entry, int hw, int chw, int classes)
{
    int coords = 4;
    int n = location / hw;
    int loc = location % hw;
    return batch * chw + n * hw * (coords + classes + 1) + entry * hw + loc;
}

static inline float logistic_activate(float x)
{
    return 1. / (1. + exp(-x));
}

static void logit_activate_array(float* x, const int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        x[i] = logistic_activate(x[i]);
    }
}

static void softmax(const float* input, int n, int stride, float* output)
{
    int i;
    float sum = 0;
    float largest = input[0];
    for (i = 0; i < n; ++i)
    {
        if (input[i * stride] > largest)
            largest = input[i * stride];
    }
    for (i = 0; i < n; ++i)
    {
        float e = exp(input[i * stride] - largest);
        sum += e;
        output[i * stride] = e;
    }
    for (i = 0; i < n; ++i)
    {
        output[i * stride] /= sum;
    }
}

static void softmax_cpu(const float* input, int n, int batch, int batch_offset, int groups, int stride, float* output)
{
    int g, b;
    for (b = 0; b < batch; ++b)
    {
        for (g = 0; g < groups; ++g)
        {
            softmax(input + b * batch_offset + g, n, stride, output + b * batch_offset + g);
        }
    }
}

static int ref_region_fp32(struct tensor* input_tensor, struct tensor* output_tensor, struct region_param* param,
                           int num_thread)
{
    int n = input_tensor->dims[0];
    int c = input_tensor->dims[1];
    int h = input_tensor->dims[2];
    int w = input_tensor->dims[3];

    int batch = n;
    int hw = h * w;
    int chw = c * hw;
    int nchw = n * chw;
    int num_box = param->num_box;
    int num_class = param->num_classes;
    int coords = param->coords;

    float* in_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    memcpy(out_data, in_data, nchw * sizeof(float));

    for (int b = 0; b < batch; b++)
    {
        for (int ni = 0; ni < num_box; ni++)
        {
            int index = entry_index(b, ni * hw, 0, hw, chw, num_class);
            logit_activate_array(out_data + index, 2 * hw);
            index = entry_index(b, ni * hw, coords, hw, chw, num_class);
            logit_activate_array(out_data + index, hw);
            index = entry_index(b, ni * hw, coords + 1, hw, chw, num_class);
        }
    }

    int index = entry_index(0, 0, coords + 1, hw, chw, num_class);
    softmax_cpu(in_data + index, num_class, batch * num_box, chw / num_box, hw, hw, out_data + index);

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
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct region_param* region_param = (struct region_param*)ir_node->op.param_mem;

    ref_region_fp32(input_tensor, output_tensor, region_param, exec_graph->num_thread);

    return 0;
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

int register_region_ref_op()
{
    return register_builtin_node_ops(OP_REGION, &hcl_node_ops);
}

int unregister_region_ref_op()
{
    return unregister_builtin_node_ops(OP_REGION, &hcl_node_ops);
}
