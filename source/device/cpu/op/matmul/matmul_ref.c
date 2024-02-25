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
 */

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

struct ref_matmul_data
{
    int batch;
    int c;
    int m;
    int n;
    int k;
    int input1_dim_num;
};

static int ref_matmul_fp32(float* input0, float* input1, float* output, struct ref_matmul_data* param)
{
    int batch = param->batch;
    int c = param->c;
    int m = param->m;
    int n = param->n;
    int k = param->k;

    for (int b = 0; b < batch; ++b)
    {
        for (int in_c = 0; in_c < c; in_c++)
        {
            float* data0 = input0 + b * c * m * n + in_c * m * n;
            float* data1 = NULL;
            switch (param->input1_dim_num)
            {
            case 4:
                data1 = input1 + b * c * n * k + in_c * n * k;
                break;
            case 3:
                data1 = input1 + in_c * n * k;
                break;
            case 2:
                data1 = input1;
                break;
            default:
                return -1;
            }
            if (data1 == NULL)
            {
                return -1;
            }
            for (int in_m = 0; in_m < m; in_m++)
            {
                //                for (int in_n = 0; in_n < n; in_n++)
                for (int in_k = 0; in_k < k; in_k++)
                {
                    float tmp = 0;
                    //                    for (int in_k = 0; in_k < k; in_k++)
                    for (int in_n = 0; in_n < n; in_n++)
                    {
                        //                        int index0 = in_m * k + in_k;
                        //                        int index1 = n * in_k + in_n;
                        int index0 = in_m * n + in_n;
                        int index1 = in_n * k + in_k;
                        tmp += data0[index0] * data1[index1];
                    }
                    *output = tmp;
                    output++;
                }
            }
        }
    }
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* input_tensor1 = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    int dim_size = input_tensor->dim_num;
    struct ref_matmul_data param;
    if (dim_size == 4)
    {
        param.batch = input_tensor->dims[0];
        param.c = input_tensor->dims[1];
        param.m = input_tensor->dims[2];
        param.n = input_tensor->dims[3];
        //        param.k = input_tensor->dims[3];
    }
    else if (dim_size == 3)
    {
        param.batch = 1;
        param.c = input_tensor->dims[0];
        param.m = input_tensor->dims[1];
        param.n = input_tensor->dims[2];
        //        param.k = input_tensor->dims[2];
    }
    else if (dim_size == 2)
    {
        param.batch = 1;
        param.c = 1; // input0->Getse().Shape(0);
        param.m = input_tensor->dims[0];
        param.n = input_tensor->dims[1];
        //        param.k = input_tensor->dims[1];
    }
    param.input1_dim_num = input_tensor1->dim_num;
    param.k = input_tensor1->dims[input_tensor1->dim_num - 1];
    void* input_data0 = input_tensor->data;
    void* input_data1 = input_tensor1->data;
    void* output_data = output_tensor->data;

    if (ref_matmul_fp32((float*)input_data0, (float*)input_data1, (float*)output_data, &param) < 0)
        return -1;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops matmul_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_matmul_ref_op()
{
    return register_builtin_node_ops(OP_MATMUL, &matmul_node_ops);
}

int unregister_matmul_ref_op()
{
    return unregister_builtin_node_ops(OP_MATMUL, &matmul_node_ops);
}
