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
 * Author: chh@openailab.com
 */

#include "batchtospacend_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>
#include <string.h>

static int ref_batchtospacend_fp32(struct tensor* input_tensor, struct tensor* output_tensor,
                                   struct batchtospacend_param* param, int num_thread)
{
    float* in_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;

    int out_dims[4];
    int in_dims[4];

    out_dims[0] = output_tensor->dims[0];
    out_dims[1] = output_tensor->dims[2];
    out_dims[2] = output_tensor->dims[3];
    out_dims[3] = output_tensor->dims[1];

    in_dims[0] = input_tensor->dims[0];
    in_dims[1] = input_tensor->dims[2];
    in_dims[2] = input_tensor->dims[3];
    in_dims[3] = input_tensor->dims[1];

    for (int in_batch = 0; in_batch < in_dims[0]; ++in_batch)
    {
        const int out_batch = (int)roundf(in_batch % out_dims[0]);
        const int spatial_offset = (int)roundf(in_batch / out_dims[0]);
        for (int in_h = 0; in_h < in_dims[1]; ++in_h)
        {
            const int out_h = (int)roundf(in_h * (param->dilation_y) + spatial_offset / (param->dilation_x) - param->crop_top);

            if (out_h < 0 || out_h >= out_dims[1])
                continue;

            for (int in_w = 0; in_w < in_dims[2]; ++in_w)
            {
                const int out_w = (int)roundf(in_w * param->dilation_x + spatial_offset % param->dilation_x - param->crop_left);

                if (out_w < 0 || out_w >= out_dims[2])
                    continue;

                int outOffset = (int)roundf(out_batch * out_dims[1] * out_dims[2] * out_dims[3] + out_h * out_dims[2] * out_dims[3] + out_w * in_dims[3]);
                float* out = out_data + outOffset;
                int inOffset = (int)roundf(in_batch * in_dims[1] * in_dims[2] * in_dims[3] + in_h * in_dims[2] * in_dims[3] + in_w * in_dims[3]);
                const float* in = in_data + inOffset;
                memcpy(out, in, in_dims[3] * sizeof(float));
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
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct batchtospacend_param* batchtospacend_param = (struct batchtospacend_param*)ir_node->op.param_mem;

    ref_batchtospacend_fp32(input_tensor, output_tensor, batchtospacend_param, exec_graph->num_thread);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_batchtospacend_ref_op()
{
    return register_builtin_node_ops(OP_BATCHTOSPACEND, &hcl_node_ops);
}

int unregister_batchtospacend_ref_op()
{
    return unregister_builtin_node_ops(OP_BATCHTOSPACEND, &hcl_node_ops);
}
