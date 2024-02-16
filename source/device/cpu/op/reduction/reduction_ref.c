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
 * Author: zpeng@openailab.com
 */

#include "reduction_param.h"

#include "reduction_kernel_ref.h"

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

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
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
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct reduction_param* reduction_param = (struct reduction_param*)ir_node->op.param_mem;
    struct reduce_param_ref param;
    int out_tensor_size = 1;
    for (int i = 0; i < output_tensor->dim_num; i++)
    {
        out_tensor_size *= output_tensor->dims[i];
    }
    int element_size = output_tensor->elem_size;

    int dims[4] = {1, 1, 1, 1};
    //    int* dims = (int*)malloc(input_tensor->dim_num * sizeof(int));
    //    memset(dims, 0, input_tensor->dim_num * sizeof(int));
    for (int i = 0; i < input_tensor->dim_num; i++)
    {
        dims[i] = input_tensor->dims[i];
    }
    int dim0 = dims[0];
    int dim1 = dims[1];
    int dim2 = dims[2];
    int dim3 = dims[3];

    param.param_dim[0] = reduction_param->dim_0;
    param.param_dim[1] = reduction_param->dim_1;
    param.param_dim[2] = reduction_param->dim_2;
    param.param_dim[3] = reduction_param->dim_3;
    for (int i = 0; i < 4; ++i)
    {
        if (param.param_dim[i] == -1)
        {
            param.param_dim[i] = input_tensor->dim_num - 1;
        }
    }
    param.type = reduction_param->type;
    int in_dim_num = input_tensor->dim_num;
    // printf("input dims: %d \n", input_tensor->dim_num);
    int ret = 0;
    if (input_tensor->data_type == TENGINE_DT_FP32)
    {
        ret = ref_reduce_fp32((float*)input_tensor->data, (float*)output_tensor->data, dim0, dim1, dim2, dim3,
                              out_tensor_size, &param, in_dim_num, dims);
    }
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
    {
        param.input_scale = input_tensor->scale;
        param.output_scale = output_tensor->scale;
        param.input_zp = input_tensor->zero_point;
        param.output_zp = output_tensor->zero_point;
        ret = ref_reduce_uint8((uint8_t*)input_tensor->data, (uint8_t*)output_tensor->data, dim0, dim1, dim2, dim3,
                               out_tensor_size, &param, in_dim_num, dims);
    }
    //    free(dims);

    return ret;
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

int register_reduction_ref_op()
{
    return register_builtin_node_ops(OP_REDUCTION, &hcl_node_ops);
}

int unregister_reduction_ref_op()
{
    return unregister_builtin_node_ops(OP_REDUCTION, &hcl_node_ops);
}
