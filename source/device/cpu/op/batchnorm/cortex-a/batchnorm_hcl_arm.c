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

#include "batchnorm_param.h"

#include "batchnorm_kernel_arm.h"

#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <math.h>
#include <string.h>

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct hcl_batchnorm_param* op_param = (struct hcl_batchnorm_param*)sys_malloc(sizeof(struct hcl_batchnorm_param));
    memset(op_param, 0, sizeof(struct hcl_batchnorm_param));
    exec_node->ops_priv = op_param;
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;

    const struct tensor* mean_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[3]);
    const struct tensor* var_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[4]);

    int channel_num = mean_tensor->dims[0];

    float* scale_mean = (float*)sys_malloc(channel_num * sizeof(float));
    float* scale_var_inv = (float*)sys_malloc(channel_num * sizeof(float));

    const float* mean = (const float*)mean_tensor->data;
    const float* var = (const float*)var_tensor->data;

    struct batchnorm_param* batchnorm_param = (struct batchnorm_param*)ir_node->op.param_mem;

    float rescale_factor;
    float eps = batchnorm_param->eps;

    rescale_factor = batchnorm_param->rescale_factor ? 1 / batchnorm_param->rescale_factor : 0;
    for (int c = 0; c < channel_num; c++)
    {
        float tmp = sqrt(var[c] * rescale_factor + eps);
        scale_var_inv[c] = (float)(1.f / tmp);
        tmp = rescale_factor * scale_var_inv[c];
        scale_mean[c] = (float)(-mean[c] * tmp);
    }
    if (!batchnorm_param->caffe_flavor)
    {
        const struct tensor* gamma_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
        const struct tensor* beta_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        const float* gamma = (const float*)gamma_tensor->data;
        const float* beta = (const float*)beta_tensor->data;
        for (int c = 0; c < channel_num; c++)
        {
            scale_var_inv[c] *= gamma[c];
            scale_mean[c] *= gamma[c];
            scale_mean[c] += beta[c];
        }
    }

    struct hcl_batchnorm_param* op_param = (struct hcl_batchnorm_param*)exec_node->ops_priv;
    op_param->scale_mean = scale_mean;
    op_param->scale_var_inv = scale_var_inv;

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

    struct hcl_batchnorm_param* op_param = (struct hcl_batchnorm_param*)exec_node->ops_priv;
    float* scale_mean = op_param->scale_mean;
    float* scale_var_inv = op_param->scale_var_inv;
    int num_thread = exec_graph->num_thread;

    batchnorm_run(output_tensor, input_tensor, scale_mean, scale_var_inv, num_thread);

    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct hcl_batchnorm_param* op_param = (struct hcl_batchnorm_param*)exec_node->ops_priv;
    sys_free(op_param->scale_mean);
    sys_free(op_param->scale_var_inv);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    if (input_tensor->data_type != TENGINE_DT_FP32 || input_tensor->layout != TENGINE_LAYOUT_NCHW)
        return 0;

    if (input_tensor->dim_num != 3 && input_tensor->dim_num != 4)
        return 0;

    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = false};

int register_batchnorm_hcl_arm_op()
{
    return register_builtin_node_ops(OP_BATCHNORM, &hcl_node_ops);
}

int unregister_batchnorm_hcl_arm_op()
{
    return unregister_builtin_node_ops(OP_BATCHNORM, &hcl_node_ops);
}
