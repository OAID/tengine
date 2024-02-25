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

#include "fc_param.h"

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

#include <math.h>
#include <string.h>

#if __SSE2__
#include <emmintrin.h>
#endif
#if __AVX__
#include <immintrin.h>
#endif

struct fc_data
{
    int need_trans;
    int batch;      // N
    int out_number; // OUT
    int hidden;     // hidden
    int zero[3];    // input, kernel, output
    float scale[3]; // input, kernel, output
};

static int innerproduct(int inn, int inc, int inh, int inw, int outc, const float* weight, const float* input, float* output,
                        const float* _bias, int num_thread, int cpu_affinity)
{
    size_t elemsize = sizeof(float);
    int size = inw * inh;
    float tmp;

    for (int n = 0; n < inn; n++)
    {
#pragma omp parallel for num_threads(num_thread)
        for (int p = 0; p < outc; p++)
        {
            int q = 0;
            float sum = _bias ? _bias[p] : 0.f;
            const float* weight1 = weight + p * inc * size;
            const float* input1 = input + n * inc * size;
#if __AVX__ || __SSE__
#if __SSE__
            float _sum[4] = {0.f};
            __m128 _sum0 = _mm_set1_ps(0.f);
            for (; q + 3 < inc * size; q = q + 4)
            {
                __m128 _input = _mm_loadu_ps(input1 + q);
                __m128 _weight = _mm_loadu_ps(weight1 + q);
                __m128 _sum1 = _mm_mul_ps(_input, _weight);
                _sum0 = _mm_add_ps(_sum0, _sum1);
            }
            _mm_storeu_ps(_sum, _sum0);
            tmp = _sum[0] + _sum[1] + _sum[2] + _sum[3];
            sum = sum + tmp;
#else //__AVX__ \
      // TODO
#endif
#endif
            for (; q < inc * size; q++)
            {
                tmp = input1[q] * weight1[q];
                sum = sum + tmp;
            }

            output[n * outc + p] = sum;
        }
    }

    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct fc_data* op_param = (struct fc_data*)sys_malloc(sizeof(struct fc_data));
    memset(op_param, 0, sizeof(struct fc_data));
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
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* output_tensor;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_param* param = (struct fc_param*)ir_node->op.param_mem;
    struct fc_data* op_param = (struct fc_data*)exec_node->ops_priv;

    if (ir_graph->graph_layout == TENGINE_LAYOUT_NCHW)
    {
        int hidden = input_tensor->dims[1];
        if (input_tensor->dim_num > 2)
            hidden = hidden * input_tensor->dims[2];
        if (input_tensor->dim_num > 3)
            hidden = hidden * input_tensor->dims[3];
        op_param->hidden = hidden;
    }
    else
    {
        int hidden = 0;
        if (input_tensor->dim_num == 2)
            hidden = input_tensor->dims[1];
        if (input_tensor->dim_num == 3)
            hidden = input_tensor->dims[1] * input_tensor->dims[2];
        if (input_tensor->dim_num == 4)
            hidden = input_tensor->dims[1] * input_tensor->dims[2] * input_tensor->dims[3];
        op_param->hidden = hidden;
    }
    op_param->batch = input_tensor->dims[0];
    op_param->out_number = param->num_output;

    int weight_out = weight_tensor->dims[0];

    if (weight_out == op_param->out_number)
        op_param->need_trans = 0;
    else
        op_param->need_trans = 1;

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* weight_tensor;
    struct tensor* bias_tensor;
    struct tensor* output_tensor;
    int num_thread = exec_graph->num_thread;
    int cpu_affinity = exec_graph->cpu_affinity;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct fc_param* param = (struct fc_param*)ir_node->op.param_mem;
    struct fc_data* op_param = (struct fc_data*)exec_node->ops_priv;

    const void* input_data = input_tensor->data;
    void* weight_data = weight_tensor->data;
    void* output_data = output_tensor->data;

    int batch_number = input_tensor->dims[0];
    int inc = input_tensor->dims[1];
    int inh = input_tensor->dims[2] ? input_tensor->dims[2] : 1;
    int inw = input_tensor->dims[3] ? input_tensor->dims[3] : 1;
    int outc = output_tensor->dims[1];

    void* bias_data = NULL;
    if (ir_node->input_num > 2)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        bias_data = bias_tensor->data;
    }
    if (innerproduct(batch_number, inc, inh, inw, outc, (float*)weight_data, (float*)input_data,
                     (float*)output_data, (float*)bias_data, num_thread, cpu_affinity)
        < 0)
        return -1;

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

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    struct node* ir_node = exec_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    /* todo support uint8 */
    if (input_tensor->data_type != TENGINE_DT_FP32)
        return 0;

    return OPS_SCORE_BEST;
}

static struct node_ops hcl_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = reshape,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_fc_hcl_x86_op()
{
    return register_builtin_node_ops(OP_FC, &hcl_node_ops);
}

int unregister_fc_hcl_x86_op()
{
    return unregister_builtin_node_ops(OP_FC, &hcl_node_ops);
}
