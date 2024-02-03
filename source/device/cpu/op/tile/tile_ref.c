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

#include "tile_param.h"

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

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int ref_tile_fp32(float* data, float* output, int* repeat, int* inDim, int* outDim, int flag)
{
    int index = 0;

    if (flag == 0) // caffe
    {
        for (int in = 0; in < inDim[0]; in++)
        {
            for (int rn = 0; rn < repeat[3]; rn++)
            {
                for (int ic = 0; ic < inDim[1]; ic++)
                {
                    for (int rc = 0; rc < repeat[2]; rc++)
                    {
                        for (int ih = 0; ih < inDim[2]; ih++)
                        {
                            for (int rh = 0; rh < repeat[1]; rh++)
                            {
                                for (int iw = 0; iw < inDim[3]; iw++)
                                {
                                    for (int rw = 0; rw < repeat[0]; rw++)
                                    {
                                        int inDataSize = in * inDim[1] * inDim[2] * inDim[3] + ic * inDim[2] * inDim[3] + ih * inDim[3] + iw;
                                        output[index] = data[inDataSize];
                                        index++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    else if (flag == 1) // onnx
    {
        int n = inDim[0];
        int c = inDim[1];
        int h = inDim[2];
        int w = inDim[3];
        int rn = repeat[3];
        int rc = repeat[2];
        int rh = repeat[1];
        int rw = repeat[0];

        int n1 = n * rn;
        int c1 = c * rc;
        int h1 = h * rh;
        int w1 = w * rw;

        int size = outDim[0] * outDim[1] * outDim[2] * outDim[3];
        for (int i = 0; i < size; ++i)
        {
            index = i / (c1 * h1 * w1) % n * (c * h * w) + i % (c1 * h1 * w1) / (h1 * w1) % c * (h * w) + i % (h1 * w1) / w1 % h * w + i % w1 % w;
            output[i] = data[index];
        }
    }

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
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* reps_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    float* input_reps = (float*)reps_tensor->data;
    const int* input_reps_shape = reps_tensor->dims;
    int* inDim = input_tensor->dims;
    int* outDim = output_tensor->dims;
    int reps_size = reps_tensor->dims[0];

    struct tile_param* param = (struct tile_param*)(ir_node->op.param_mem);

    int frame_flag = param->frame_flag;
    struct vector* repeat = create_vector(sizeof(int), NULL);
    int size = 0;
    int default_value = 1;

    if (frame_flag == 0)
    {
        size = param->reps_size;
        for (int i = 0; i < 4 - size; i++)
        {
            push_vector_data(repeat, (void*)&default_value);
        }
    }
    else if (frame_flag == 1)
    {
        size = input_reps_shape[0] * input_reps_shape[1] * input_reps_shape[2] * input_reps_shape[3];
        for (int i = 0; i < size; i++)
        {
            push_vector_data(repeat, (void*)&input_reps[i]);
        }
        for (int i = 0; i < 4 - size; i++)
        {
            push_vector_data(repeat, (void*)&default_value);
        }
    }

    int* repeat_data = (int*)sys_malloc(get_vector_num(repeat) * sizeof(int));
    for (int i = 0; i < get_vector_num(repeat); i++)
    {
        int* a = (int*)get_vector_data(repeat, i);
        repeat_data[i] = *a;
    }

    ref_tile_fp32((float*)input_tensor->data, (float*)output_tensor->data, repeat_data, inDim, outDim, frame_flag);
    sys_free(repeat_data);
    release_vector(repeat);

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
    .is_ref_op = true};

int register_tile_ref_op()
{
    return register_builtin_node_ops(OP_TILE, &hcl_node_ops);
}

int unregister_tile_ref_op()
{
    return unregister_builtin_node_ops(OP_TILE, &hcl_node_ops);
}
