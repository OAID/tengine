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
 * Author: jxyang@openailab.com
 * Update: hhchen@openailab.com
 */

#include "gather_param.h"

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

typedef struct
{
    int* in_shape;  // the dim of the input
    int* out_shape; // dims of output
    int out_dim_num;
    int axis;
    int indices_num;
    int dim_size;
    int is_onnx;
} gather_param_t;

static int ref_gather_fp32(float* input, int* input_indices, float* output, gather_param_t* param, int num_thread)
{
    float* out_ptr = output;
    float* in_ptr = input;
    int axis = param->axis;
    int outer_size = 1;
    int inner_size = 1;
    int axis_size = param->in_shape[axis];

    for (int i = 0; i < axis; i++)
    {
        outer_size *= param->in_shape[i];
    }

    for (int i = axis + 1; i < param->dim_size; i++)
    {
        inner_size *= param->in_shape[i];
        // TLOG_ERR("inner_size size: %d %d \n", inner_size, param->in_shape[i]);
    }

    int out_n, out_c, out_h, out_w;
    out_n = out_c = out_h = out_w = 1;
    if (param->out_dim_num == 4)
    {
        out_n = param->out_shape[0];
        out_c = param->out_shape[1];
        out_h = param->out_shape[2];
        out_w = param->out_shape[3];
    }
    else if (param->out_dim_num == 3)
    {
        out_c = param->out_shape[0];
        out_h = param->out_shape[1];
        out_w = param->out_shape[2];
    }
    else if (param->out_dim_num == 2)
    {
        out_h = param->out_shape[0];
        out_w = param->out_shape[1];
    }
    else
        return -1;

    // #pragma omp parallel for num_threads(num_thread)
    if (param->is_onnx)
    {
        for (int n = 0; n < out_n; ++n)
        {
            for (int c = 0; c < out_c; ++c)
            {
                for (int h = 0; h < out_h; ++h)
                {
                    for (int w = 0; w < out_w; ++w)
                    {
                        int indices_i;
                        int input_id;
                        switch (axis)
                        {
                        case 0:
                            indices_i = input_indices[n];
                            input_id = indices_i * out_c * out_h * out_w + c * out_h * out_w + h * out_w + w;
                            break;
                        case 1:
                            indices_i = input_indices[c];
                            input_id = n * out_c * out_h * out_w + indices_i * out_h * out_w + h * out_w + w;
                            break;
                        case 2:
                            indices_i = input_indices[h];
                            input_id = n * out_c * out_h * out_w + c * out_h * out_w + indices_i * out_w + w;
                            break;
                        case 3:
                            indices_i = input_indices[w];
                            input_id = n * out_c * out_h * out_w + c * out_h * out_w + h * out_w + indices_i;
                            break;
                        default:
                            return -1;
                        }
                        int output_id = n * out_c * out_h * out_w + c * out_h * out_w + h * out_w + w;
                        output[output_id] = input[input_id];
                    }
                }
            }
        }
    }
    else
    {
        for (int outer = 0; outer < outer_size; ++outer)
        {
            for (int i = 0; i < param->indices_num; i++)
            {
                memcpy(out_ptr + (outer * param->indices_num + i) * inner_size,
                       in_ptr + (outer * axis_size + (int)input_indices[i]) * inner_size, inner_size * sizeof(float));
            }
        }
    }
    return 0;
}

static int ref_gather_uint8(uint8_t* input, int* input_indices, uint8_t* output, gather_param_t* param, int num_thread)
{
    uint8_t* out_ptr = output;
    uint8_t* in_ptr = input;
    int axis = param->axis;
    int outer_size = 1;
    int inner_size = 1;
    int axis_size = param->in_shape[axis];

    for (int i = 0; i < axis; i++)
    {
        outer_size *= param->in_shape[i];
    }

    for (int i = axis + 1; i < param->dim_size; i++)
    {
        inner_size *= param->in_shape[i];
    }

    // #pragma omp parallel for num_threads(num_thread)
    for (int outer = 0; outer < outer_size; ++outer)
    {
        for (int i = 0; i < param->indices_num; i++)
        {
            memcpy(out_ptr + (outer * param->indices_num + i) * inner_size,
                   in_ptr + (outer * axis_size + (int)input_indices[i]) * inner_size, inner_size);
        }
    }

    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct gather_param* gather_param = (struct gather_param*)ir_node->op.param_mem;
    gather_param_t* op_priv_info = (gather_param_t*)exec_node->ops_priv;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    op_priv_info->axis = gather_param->axis;
    op_priv_info->indices_num = gather_param->indices_num;
    op_priv_info->is_onnx = gather_param->is_onnx;
    op_priv_info->in_shape = (int*)sys_malloc(input_tensor->dim_num * sizeof(int));
    op_priv_info->out_shape = (int*)sys_malloc(output_tensor->dim_num * sizeof(int));
    /* prerun now */
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct tensor* indices_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    gather_param_t* op_priv_info = (gather_param_t*)exec_node->ops_priv;

    int out_size = input_tensor->elem_num;

    // auto in_dim = input_tensor->GetShape().GetDim();
    void* input = input_tensor->data;

    void* indices_data = indices_tensor->data;

    op_priv_info->dim_size = input_tensor->dim_num;
    for (int i = 0; i < op_priv_info->dim_size; i++)
    {
        op_priv_info->in_shape[i] = input_tensor->dims[i];
    }
    op_priv_info->out_dim_num = output_tensor->dim_num;
    for (int i = 0; i < op_priv_info->out_dim_num; ++i)
    {
        op_priv_info->out_shape[i] = output_tensor->dims[i];
    }
    // TLOG_ERR("in shape: %d %d %d %d\n", op_priv_info->in_shape[0], op_priv_info->in_shape[1], op_priv_info->in_shape[3], op_priv_info->in_shape[3]);

    // int indices_num = op_param.indices_num;
    void* output = output_tensor->data;

    int ret = -1;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_gather_fp32((float*)input, (int*)indices_data, (float*)output, op_priv_info, exec_graph->num_thread);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_gather_uint8((uint8_t*)input, (int*)indices_data, (uint8_t*)output, op_priv_info, exec_graph->num_thread);

    return ret;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;

    gather_param_t* op_priv_info = (gather_param_t*)sys_malloc(sizeof(gather_param_t));

    if (op_priv_info == NULL)
    {
        return -1;
    }

    memset(op_priv_info, 0, sizeof(gather_param_t));

    exec_node->ops_priv = op_priv_info;

    return 0;
}
static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    gather_param_t* op_param = (gather_param_t*)exec_node->ops_priv;

    sys_free(op_param->in_shape);
    sys_free(op_param->out_shape);

    return 0;
}
static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    gather_param_t* op_priv_info = (gather_param_t*)exec_node->ops_priv;

    sys_free(op_priv_info);

    exec_node->ops_priv = NULL;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops gather_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_gather_ref_op()
{
    return register_builtin_node_ops(OP_GATHER, &gather_node_ops);
}

int unregister_gather_ref_op()
{
    return unregister_builtin_node_ops(OP_GATHER, &gather_node_ops);
}
