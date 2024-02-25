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
 * Author: bhu@openailab.com
 * Update: hhchen@openailab.com
 */

#include "batchnorm_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <string.h>

#include "batchnorm_kernel_ref.h"

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ref_batchnorm_param* batchnorm_op_param = (struct ref_batchnorm_param*)sys_malloc(sizeof(struct ref_batchnorm_param));
    memset(batchnorm_op_param, 0, sizeof(struct ref_batchnorm_param));
    exec_node->ops_priv = batchnorm_op_param;
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
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct tensor* mean_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[3]);
    struct tensor* var_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[4]);

    struct ref_batchnorm_param* op_param = (struct ref_batchnorm_param*)exec_node->ops_priv;
    struct batchnorm_param* batchnorm_param = (struct batchnorm_param*)ir_node->op.param_mem;

    int channel_num = input_tensor->dims[1];

    float* scale_mean = (float*)sys_malloc(channel_num * sizeof(float));
    float* scale_var_inv = (float*)sys_malloc(channel_num * sizeof(float));
    const float* mean = (const float*)mean_tensor->data;
    const float* var = (const float*)var_tensor->data;

    float rescale_factor;
    float eps = batchnorm_param->eps;

    rescale_factor = batchnorm_param->rescale_factor ? 1 / batchnorm_param->rescale_factor : 0;

    for (int c = 0; c < channel_num; c++)
    {
        float tmp = sqrtf(var[c] * rescale_factor + eps);
        scale_var_inv[c] = (float)(1.f / tmp);
        tmp = rescale_factor * scale_var_inv[c];
        scale_mean[c] = (float)(-mean[c] * tmp);
    }
    float* gamma = NULL;
    float* beta = NULL;
    if (!batchnorm_param->caffe_flavor)
    {
        const struct tensor* gamma_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
        const struct tensor* beta_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        gamma = (float*)gamma_tensor->data;
        beta = (float*)beta_tensor->data;
    }
    int layout = ir_graph->graph_layout;
    op_param->iscaffe = batchnorm_param->caffe_flavor;
    op_param->scale_mean = scale_mean;
    op_param->scale_var_inv = scale_var_inv;
    op_param->gamma = gamma;
    op_param->beta = beta;
    op_param->layout = layout;

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct ref_batchnorm_param* batchnorm_op_param = (struct ref_batchnorm_param*)exec_node->ops_priv;
    void* out_data = output_tensor->data;
    void* input = input_tensor->data;

    if (4 == input_tensor->dim_num)
    {
        batchnorm_op_param->input_n = input_tensor->dims[0];
        batchnorm_op_param->input_c = input_tensor->dims[1];
        batchnorm_op_param->input_h = input_tensor->dims[2];
        batchnorm_op_param->input_w = input_tensor->dims[3];
    }
    else if (3 == input_tensor->dim_num)
    {
        batchnorm_op_param->input_n = input_tensor->dims[0];
        batchnorm_op_param->input_c = input_tensor->dims[1];
        batchnorm_op_param->input_w = input_tensor->dims[2];
        batchnorm_op_param->input_h = 1;
    }
    else if (2 == input_tensor->dim_num)
    {
        batchnorm_op_param->input_n = input_tensor->dims[0];
        batchnorm_op_param->input_c = input_tensor->dims[1];
        batchnorm_op_param->input_h = 1;
        batchnorm_op_param->input_w = 1;
    }
    else
    {
        return -1;
    }

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_batchnorm_fp32((float*)input, (float*)out_data, batchnorm_op_param);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_batchnorm_uint8(input_tensor, output_tensor, batchnorm_op_param);

    return ret;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ref_batchnorm_param* batchnorm_op_param = (struct ref_batchnorm_param*)exec_node->ops_priv;

    sys_free(batchnorm_op_param->scale_mean);
    sys_free(batchnorm_op_param->scale_var_inv);

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
    .postrun = postrun,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_batchnorm_ref_op()
{
    return register_builtin_node_ops(OP_BATCHNORM, &hcl_node_ops);
}

int unregister_batchnorm_ref_op()
{
    return unregister_builtin_node_ops(OP_BATCHNORM, &hcl_node_ops);
}
