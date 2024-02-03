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
 * Author: haitao@openailab.com
 */

#include "pooling_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "utility/float.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include "pooling_kernel_ref.h"

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct pool_param* pool_param = (struct pool_param*)ir_node->op.param_mem;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_pooling_fp32(input_tensor, output_tensor, pool_param, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_FP16)
#if MACOS
        TLOG_ERR("FP16 not support mac os");
#else
        ret = ref_pooling_fp16(input_tensor, output_tensor, pool_param, exec_graph->num_thread);
#endif
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_pooling_uint8(input_tensor, output_tensor, pool_param, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_pooling_int8(input_tensor, output_tensor, pool_param, exec_graph->num_thread);
    else
        TLOG_ERR("Input data type %d not to be supported.\n", input_tensor->data_type);

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct pool_param* pool_param = (struct pool_param*)ir_node->op.param_mem;

    int ret = 0;

    int batch = input_tensor->dims[0];
    int channel = input_tensor->dims[1];
    int input_h = input_tensor->dims[2];
    int input_w = input_tensor->dims[3];
    int output_h, output_w;

    if (pool_param->kernel_h == input_h && pool_param->kernel_w == input_w && pool_param->pad_w0 == 0 && pool_param->pad_w1 == 0 && pool_param->pad_h0 == 0 && pool_param->pad_h1 == 0)
    {
        pool_param->global = 1;
    }

    if (pool_param->global)
    {
        pool_param->pad_h0 = 0;
        pool_param->pad_h1 = 0;
        pool_param->pad_w0 = 0;
        pool_param->pad_w1 = 0;
        pool_param->kernel_h = input_h;
        pool_param->kernel_w = input_w;
        pool_param->pad_h0 = pool_param->pad_h1 = pool_param->pad_w0 = pool_param->pad_w1 = 0;
        pool_param->stride_h = pool_param->stride_w = 1;
        output_h = 1;
        output_w = 1;
    }
    else
    {
        int caffe = pool_param->caffe_flavor & ~(COUNT_INCLUDE_PAD_MSK);
        output_h = calc_output_size(input_h, pool_param->kernel_h, pool_param->stride_h, pool_param->pad_h0_org,
                                    pool_param->caffe_flavor);
        output_w = calc_output_size(input_w, pool_param->kernel_w, pool_param->stride_w, pool_param->pad_w0_org,
                                    pool_param->caffe_flavor);
        if (2 != caffe)
        {
            calc_real_pads(output_h, input_h, pool_param->kernel_h, pool_param->stride_h, pool_param->pad_h0_org,
                           &pool_param->pad_h0, &pool_param->pad_h1);
            calc_real_pads(output_w, input_w, pool_param->kernel_w, pool_param->stride_w, pool_param->pad_w0_org,
                           &pool_param->pad_w0, &pool_param->pad_w1);
        }
        else
        {
            int pad_w0 = pool_param->pad_w0_org;
            int pad_h0 = pool_param->pad_h0_org;
            pool_param->pad_w0 = pad_w0 / 2;
            pool_param->pad_h0 = pad_h0 / 2;
            pool_param->pad_w1 = pad_w0 - pad_w0 / 2;
            pool_param->pad_h1 = pad_h0 - pad_h0 / 2;
        }
    }

    int dims[4];

    if (output_tensor->dims[1] != channel || output_tensor->dims[2] != output_h || output_tensor->dims[3] != output_w)
    {
        dims[0] = batch;
        dims[1] = channel;
        dims[2] = output_h;
        dims[3] = output_w;
        ret = set_ir_tensor_shape(output_tensor, dims, 4);
    }

    return ret;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* dev)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* dev)
{
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = NULL,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_pooling_ref_op()
{
    return register_builtin_node_ops(OP_POOL, &hcl_node_ops);
}

int unregister_pooling_ref_op()
{
    unregister_builtin_node_ops(OP_POOL, &hcl_node_ops);
    return 0;
}
