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
 * Author: haoluo@openailab.com
 */

#include "deconv_param.h"

#include "cortex_a/deconv_kernel_arm.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* filter_tensor;
    struct tensor* output_tensor;

    struct deconv_priv_info* deconv_priv_info = (struct deconv_priv_info*)exec_node->ops_priv;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct deconv_param* deconv_param = (struct deconv_param*)ir_node->op.param_mem;

    /* prerun now */
    if (deconv_hcl_prerun(input_tensor, filter_tensor, output_tensor, deconv_priv_info, deconv_param) < 0)
    {
        TLOG_ERR("hcl deconv prerun failed\n");
        // set_tengine_errno(EFAULT);
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* bias_tensor = NULL;
    struct tensor* output_tensor = NULL;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct deconv_param* deconv_param = (struct deconv_param*)ir_node->op.param_mem;
    struct deconv_priv_info* deconv_priv_info = (struct deconv_priv_info*)exec_node->ops_priv;

    if (deconv_hcl_run(input_tensor, weight_tensor, bias_tensor, output_tensor, deconv_priv_info, deconv_param,
                       num_thread, cpu_affinity)
        < 0)
    {
        TLOG_ERR("hcl deconv run failed\n");
        // set_tengine_errno(EFAULT);
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
    struct deconv_priv_info* deconv_priv_info = (struct deconv_priv_info*)exec_node->ops_priv;

    if (deconv_hcl_postrun(deconv_priv_info) < 0)
    {
        TLOG_ERR("hcl deconv prerun failed\n");
        // set_tengine_errno(EFAULT);
        return -1;
    }
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct deconv_param* deconv_param = (struct deconv_param*)ir_node->op.param_mem;
    struct deconv_priv_info* deconv_priv_info = (struct deconv_priv_info*)sys_malloc(sizeof(struct deconv_priv_info));

    if (deconv_priv_info == NULL)
    {
        // set_tengine_errno(ENOMEM);
        return -1;
    }

    memset(deconv_priv_info, 0, sizeof(struct deconv_priv_info));
    exec_node->ops_priv = deconv_priv_info;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct deconv_priv_info* deconv_priv_info = (struct deconv_priv_info*)exec_node->ops_priv;
    sys_free(deconv_priv_info);
    exec_node->ops_priv = NULL;
    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
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

int register_deconv_hcl_arm_op()
{
    return register_builtin_node_ops(OP_DECONV, &hcl_node_ops);
}

int unregister_deconv_hcl_arm_op()
{
    unregister_builtin_node_ops(OP_DECONV, &hcl_node_ops);
    return 0;
}
