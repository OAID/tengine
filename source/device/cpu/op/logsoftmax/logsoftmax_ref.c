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

#include "logsoftmax_param.h"

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

struct ref_logsoftmax_param
{
    int axis;
    int in_size;
    int on_size;
    int out_size;
    float scale[2];    // scale[0]: input scale, scale[1]: output scale
    int zero_point[2]; // zero_point[0]: input zero_point, zero_point[1]: output zero_point
};

static void GetMaxArray(float* input, float* array, int in_size, int on_size)
{
    float* input_ptr = (float*)input;
    float* array_ptr = (float*)array;
    memset(array, 0, in_size * sizeof(float));

    for (int j = 0; j < on_size; j++)
        for (int l = 0; l < in_size; l++)
        {
            if (array_ptr[l] < input_ptr[j * in_size + l])
                array_ptr[l] = input_ptr[j * in_size + l];
        }
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ref_logsoftmax_param* logsoftmax_op_param = (struct ref_logsoftmax_param*)sys_malloc(sizeof(struct ref_logsoftmax_param));
    memset(logsoftmax_op_param, 0, sizeof(struct ref_logsoftmax_param));
    exec_node->ops_priv = logsoftmax_op_param;
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    return 0;
}

static void GetOutResult(float* input, float* output, float* array, float* sum_array, int in_size, int on_size)
{
    float* input_ptr = (float*)input;
    float* output_ptr = (float*)output;
    float* array_ptr = (float*)array;
    float* sum_array_ptr = (float*)sum_array;

    memset(sum_array, 0x0, in_size * sizeof(float));

    /* get the exp and the summary */

    for (int j = 0; j < on_size; j++)
        for (int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] = exp(input_ptr[index] - array_ptr[l]);
            sum_array_ptr[l] += output_ptr[index];
        }

    /* the final result */
    for (int j = 0; j < on_size; j++)
        for (int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] /= sum_array_ptr[l];
            output_ptr[index] = log(output_ptr[index]);
        }
}

static int ref_logsoftmax_fp32(float* input_data, float* output_data, float* max_array, float* sum_array, struct ref_logsoftmax_param* op_param)
{
    for (int i = 0; i < op_param->out_size; i++)
    {
        int img_base = i * op_param->in_size * op_param->on_size;
        GetMaxArray(input_data + img_base, max_array, op_param->in_size, op_param->on_size);
        GetOutResult(input_data + img_base, output_data + img_base, max_array, sum_array, op_param->in_size, op_param->on_size);
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
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    struct ref_logsoftmax_param ref_logsoftmax_param;
    float* max_array;
    float* sum_array;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    ref_logsoftmax_param.out_size = input_tensor->elem_num;
    ref_logsoftmax_param.scale[0] = input_tensor->scale;
    ref_logsoftmax_param.scale[1] = output_tensor->scale;
    ref_logsoftmax_param.zero_point[0] = input_tensor->zero_point;
    ref_logsoftmax_param.zero_point[1] = output_tensor->zero_point;

    struct logsoftmax_param* param_ = (struct logsoftmax_param*)(ir_node->op.param_mem);

    // const std::vector<int>& dims = input_tensor->GetShape().GetDim();
    const int* dims = input_tensor->dims;
    //
    int axis = param_->axis;
    int out_size = 1;
    for (int i = 0; i < axis; i++)
    {
        out_size *= dims[i];
    }
    int in_size = 1;
    for (size_t i = axis + 1; i < input_tensor->dim_num; i++)
    {
        in_size *= dims[i];
    }
    int on_size = dims[axis];

    max_array = (float*)sys_malloc(in_size * sizeof(float));
    sum_array = (float*)sys_malloc(in_size * sizeof(float));

    ref_logsoftmax_param.in_size = in_size;
    ref_logsoftmax_param.on_size = on_size;

    if (input_tensor->data_type == TENGINE_DT_FP32)
        ref_logsoftmax_fp32((float*)input_tensor->data, (float*)output_tensor->data, max_array, sum_array, &ref_logsoftmax_param);
    // else
    // ref_logistic_uint8(input_tensor->data, output_tensor->data, &logical_param);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {.prerun = prerun,
                                       .run = run,
                                       .reshape = NULL,
                                       .postrun = NULL,
                                       .init_node = init_node,
                                       .release_node = release_node,
                                       .score = score,
                                       .is_ref_op = true};

int register_logsoftmax_ref_op()
{
    return register_builtin_node_ops(OP_LOGSOFTMAX, &hcl_node_ops);
}

int unregister_logsoftmax_ref_op()
{
    return unregister_builtin_node_ops(OP_LOGSOFTMAX, &hcl_node_ops);
}
