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
 * update: qtang@openailab.com
 */

#include "fc_param.h"

#include "fc_kernel_arm.h"
#include "fc_kernel_int8_arm.h"

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

#include <string.h>

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "cortex_a/fc_kernel_fp16_arm82.h"
#endif

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* filter_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_priv_info* priv_info = (struct fc_priv_info*)exec_node->ops_priv;
    struct fc_param* fc_param = (struct fc_param*)ir_node->op.param_mem;

    /* fp32 prerun */
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        if (fc_kernel_prerun(input_tensor, filter_tensor, output_tensor, priv_info, fc_param) < 0)
        {
            TLOG_ERR("hcl fc prerun failed\n");
            return -1;
        }
    }
    /* fp16 prerun */
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else if (exec_graph->mode == TENGINE_MODE_FP16)
    {
        if (fp16_fc_kernel_prerun(input_tensor, filter_tensor, output_tensor, priv_info, fc_param) < 0)
        {
            TLOG_ERR("hcl fp16 fc prerun failed\n");
            // set_tengine_errno(EFAULT);
            return -1;
        }
    }
#endif
    else if (exec_graph->mode == TENGINE_MODE_INT8)
    {
        if (int8_fc_kernel_prerun(input_tensor, filter_tensor, output_tensor, priv_info, fc_param) < 0)
        {
            TLOG_ERR("hcl fc prerun failed\n");
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
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* bias_tensor = NULL;
    struct tensor* output_tensor = NULL;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    /* set the input data and shape again, in case of reshape or dynamic shape */
    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    if (ir_node->input_num > 2)
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_param* fc_param = (struct fc_param*)ir_node->op.param_mem;
    struct fc_priv_info* priv_info = (struct fc_priv_info*)exec_node->ops_priv;

    /* fp32 run */
    if (exec_graph->mode == TENGINE_MODE_FP32)
    {
        if (fc_kernel_run(input_tensor, weight_tensor, bias_tensor, output_tensor, priv_info, fc_param, num_thread,
                          cpu_affinity)
            < 0)
        {
            TLOG_ERR("hcl fc run failed\n");
            return -1;
        }
    }
    /* fp16 run */
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    else if (exec_graph->mode == TENGINE_MODE_FP16)
    {
        if (fp16_fc_kernel_run(input_tensor, weight_tensor, bias_tensor, output_tensor, priv_info, fc_param, num_thread, cpu_affinity) < 0)
        {
            TLOG_ERR("hcl fp16 fc run failed\n");
            // set_tengine_errno(EFAULT);
            return -1;
        }
    }
#endif
    else if (exec_graph->mode == TENGINE_MODE_INT8)
    {
        if (int8_fc_kernel_run(input_tensor, weight_tensor, bias_tensor, output_tensor, priv_info, fc_param, num_thread, cpu_affinity) < 0)
        {
            TLOG_ERR("hcl fc run failed\n");
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
    struct node* node = exec_node->ir_node;
    struct graph* graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);

    int dim[4];

    int n = weight->dims[0];
    int k = weight->dims[1];

    int m = input->dims[0];
    int input_k = input->dims[1];

    if (input->dim_num == 2)
    {
        dim[0] = m;
        dim[1] = n;
    }
    else if (input->dim_num == 3)
    {
        if (input->dims[2] != 0)
            input_k *= input->dims[2];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
        }
    }
    else if (input->dim_num == 4)
    {
        if (input->dims[2] * input->dims[3] != 0)
            input_k *= input->dims[2] * input->dims[3];
        if (graph->graph_layout == TENGINE_LAYOUT_NHWC)
        {
            dim[0] = m;
            dim[1] = 1;
            dim[2] = 1;
            dim[3] = n;
        }
        else
        {
            dim[0] = m;
            dim[1] = n;
            dim[2] = 1;
            dim[3] = 1;
        }
    }
    else
        return -1;

    if (k != input_k)
    {
        TLOG_ERR("fc: input tensor and weight tensor shape does not match, hidden_number: %d\n", k);
        return -1;
    }

    int ret = set_ir_tensor_shape(output, dim, input->dim_num);

    return ret;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct fc_priv_info* priv_info = (struct fc_priv_info*)exec_node->ops_priv;

    /* fp32 postrun */
    if (exec_graph->mode == TENGINE_MODE_FP32 || exec_graph->mode == TENGINE_MODE_UINT8)
    {
        if (fc_kernel_postrun(priv_info) < 0)
        {
            TLOG_ERR("hcl fc postrun failed\n");
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
    /* init the private info data of convolution op */
    struct fc_priv_info* priv_info = (struct fc_priv_info*)sys_malloc(sizeof(struct fc_priv_info));
    if (priv_info == NULL)
    {
        return -1;
    }
    memset(priv_info, 0, sizeof(struct fc_priv_info));
    exec_node->ops_priv = priv_info;

    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct fc_priv_info* priv_info = (struct fc_priv_info*)exec_node->ops_priv;
    sys_free(priv_info);
    exec_node->ops_priv = NULL;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    /* todo support uint8 */
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    if (input_tensor->data_type != TENGINE_DT_FP32 && input_tensor->data_type != TENGINE_DT_FP16)
        return 0;
#else
    if (input_tensor->data_type != TENGINE_DT_FP32
        // && input_tensor->data_type != TENGINE_DT_INT8    // 从tengine移植的 fc int8 arm 与 fc int8 ref 相比相差较大，暂且关闭
    )
        return 0;
#endif

    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = reshape,
                                       .postrun = postrun,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = false};

int register_fc_hcl_arm_op()
{
    return register_builtin_node_ops(OP_FC, &hcl_node_ops);
}

int unregister_fc_hcl_arm_op()
{
    unregister_builtin_node_ops(OP_FC, &hcl_node_ops);
    return 0;
}
