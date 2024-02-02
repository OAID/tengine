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

#include "convolution_param.h"

#include "conv_kernel_mips.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "module/module.h"
#include "operator/op.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = (struct conv_priv_info*)exec_node->ops_priv;

    /* get cpu affinity */
    conv_priv_info->cpu_type = exec_graph->cpu_affinity;

    /* fp32 prerun */
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        if (conv_hcl_set_shared_mem && exec_node->shared_mem_size < exec_graph->shared_mem_size)
        {
            if (conv_hcl_set_shared_mem(conv_priv_info, exec_graph->shared_mem, exec_graph->shared_mem_size) < 0)
            {
                TLOG_ERR("hcl conv: set shared memory failed\n");
                // set_tengine_errno(EFAULT);
                return -1;
            }
        }
        if (conv_hcl_set_shared_pack4_mem && exec_node->shared_pack4_mem_size < exec_graph->shared_pack4_mem_size)
        {
            if (conv_hcl_set_shared_pack4_mem(conv_priv_info, exec_graph->shared_pack4_mem,
                                              exec_graph->shared_pack4_mem_size)
                < 0)
            {
                TLOG_ERR("hcl conv: set shared pack4 memory failed\n");
                // set_tengine_errno(EFAULT);
                return -1;
            }
        }

        int group = conv_param->group;
        int kernel_h = conv_param->kernel_h;
        int kernel_w = conv_param->kernel_w;
        if (group > 1 && kernel_h == 7 && kernel_w == 7)
            conv_priv_info->external_interleave_pack4_mem = 0;
        else
            conv_priv_info->external_interleave_pack4_mem = 1;

        /* do prerun */
        if (conv_hcl_prerun(input_tensor, filter_tensor, output_tensor, conv_priv_info, conv_param) < 0)
        {
            TLOG_ERR("hcl conv prerun failed\n");
            // set_tengine_errno(EFAULT);
            return -1;
        }
    }
    else
    {
        TLOG_ERR("Tengine work node not support %d\n", exec_graph->mode);
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    // TLOG_ERR("conv hcl start\n");
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* output_tensor;
    struct tensor* bias_tensor = NULL;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);

    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;
    struct conv_priv_info* conv_priv_info = (struct conv_priv_info*)exec_node->ops_priv;

    /* fp32 run */
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        if (conv_hcl_run(input_tensor, weight_tensor, bias_tensor, output_tensor, conv_priv_info, conv_param, num_thread,
                         cpu_affinity)
            < 0)
        {
            TLOG_ERR("hcl conv run failed\n");
            // set_tengine_errno(EFAULT);
            return -1;
        }
    }
    else
    {
        TLOG_ERR("Tengine work node not support %d\n", exec_graph->mode);
        return -1;
    }

    return 0;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* conv_priv_info = (struct conv_priv_info*)exec_node->ops_priv;

    /* fp32 postrun */
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        if (conv_hcl_postrun(conv_priv_info) < 0)
        {
            TLOG_ERR("hcl conv postrun failed\n");
            // set_tengine_errno(EFAULT);
            return -1;
        }
    }
    else
    {
        TLOG_ERR("Tengine work node not support %d\n", exec_graph->mode);
        return -1;
    }

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* filter_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;

    /* init the private info data of convolution op */
    struct conv_priv_info* conv_priv_info = (struct conv_priv_info*)sys_malloc(sizeof(struct conv_priv_info));
    if (conv_priv_info == NULL)
    {
        // set_tengine_errno(ENOMEM);
        return -1;
    }
    memset(conv_priv_info, 0, sizeof(struct conv_priv_info));
    exec_node->ops_priv = conv_priv_info;

    /* get shared memory size */
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        exec_node->shared_mem_size = conv_hcl_get_shared_mem_size(input_tensor, output_tensor, conv_param);
        exec_node->shared_pack4_mem_size = conv_hcl_get_shared_pack4_mem_size(filter_tensor, output_tensor, conv_param);
    }
    else
    {
        TLOG_ERR("Tengine work node not support %d\n", exec_graph->mode);
        return -1;
    }

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct conv_priv_info* conv_priv_info = (struct conv_priv_info*)exec_node->ops_priv;
    sys_free(conv_priv_info);
    exec_node->ops_priv = NULL;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct conv_param* param = (struct conv_param*)exec_node->op.param_mem;
    int group = param->group;
    int kernel_h = param->kernel_h;
    int kernel_w = param->kernel_w;
    int in_c = input_tensor->dims[1] / group;
    int out_c = output_tensor->dims[1] / group;

    if (input_tensor->data_type != TENGINE_DT_FP32)
        return 0;

    if (group != 1)
        return 0;

    return OPS_SCORE_PREFER;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = false};

int register_conv_hcl_mips_op()
{
    return register_builtin_node_ops(OP_CONV, &hcl_node_ops);
}

int unregister_conv_hcl_mips_op()
{
    unregister_builtin_node_ops(OP_CONV, &hcl_node_ops);
    return 0;
}
