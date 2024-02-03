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

#include "depthtospace_param.h"

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

int ref_depthtospace_fp32(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread, int block_size)
{
    int n = input_tensor->dims[0];
    int inc = input_tensor->dims[1];
    int inh = input_tensor->dims[2];
    int inw = input_tensor->dims[3];

    int outc = inc / (block_size * block_size);
    int outh = input_tensor->dims[2] * block_size;
    int outw = input_tensor->dims[3] * block_size;

    float* input_data = (float*)input_tensor->data;
    float* out_data = (float*)output_tensor->data;
    int total_size = input_tensor->elem_num;

    //TODO:add mode in depthtospace_param to set CRD or DCR
    for (int b = 0; b < n; ++b)
    {
        for (int s = 0; s < outc; ++s)
        {
            for (int h = 0; h < outh; ++h)
            {
                const int in_h = h / block_size;
                const int offset_h = (h % block_size);
                for (int w = 0; w < outw; ++w)
                {
                    const int in_w = w / block_size;
                    const int offset_w = w % block_size;
                    //CRD
                    const int offset_d = offset_h * block_size + offset_w;
                    const int in_d = s * (block_size * block_size) + offset_d;
                    //DCR
                    //const int offset_d =(offset_h * block_size + offset_w) * outc;
                    //const int in_d = s + offset_d;
                    const int o_index = ((b * outc + s) * outh + h) * outw + w;
                    const int i_index = ((b * inc + in_d) * inh + in_h) * inw + in_w;
                    out_data[o_index] = input_data[i_index];
                }
            }
        }
    }

    return 0;
}

int ref_depthtospace_uint8(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread, int block_size)
{
    int n = input_tensor->dims[0];
    int inc = input_tensor->dims[1];
    int inh = input_tensor->dims[2];
    int inw = input_tensor->dims[3];

    int outc = inc / (block_size * block_size);
    int outh = input_tensor->dims[2] * block_size;
    int outw = input_tensor->dims[3] * block_size;

    unsigned char* input_data = (unsigned char*)input_tensor->data;
    unsigned char* out_data = (unsigned char*)output_tensor->data;
    int total_size = input_tensor->elem_num;

    //TODO:add mode in depthtospace_param to set CRD or DCR
    for (int b = 0; b < n; ++b)
    {
        for (int s = 0; s < outc; ++s)
        {
            for (int h = 0; h < outh; ++h)
            {
                const int in_h = h / block_size;
                const int offset_h = (h % block_size);
                for (int w = 0; w < outw; ++w)
                {
                    const int in_w = w / block_size;
                    const int offset_w = w % block_size;
                    //CRD
                    const int offset_d = offset_h * block_size + offset_w;
                    const int in_d = s * (block_size * block_size) + offset_d;
                    //DCR
                    //const int offset_d =(offset_h * block_size + offset_w) * outc;
                    //const int in_d = s + offset_d;
                    const int o_index = ((b * outc + s) * outh + h) * outw + w;
                    const int i_index = ((b * inc + in_d) * inh + in_h) * inw + in_w;
                    out_data[o_index] = input_data[i_index];
                }
            }
        }
    }

    return 0;
}

int ref_depthtospace_int8(struct tensor* input_tensor, struct tensor* output_tensor, int num_thread, int block_size)
{
    int n = input_tensor->dims[0];
    int inc = input_tensor->dims[1];
    int inh = input_tensor->dims[2];
    int inw = input_tensor->dims[3];

    int outc = inc / (block_size * block_size);
    int outh = input_tensor->dims[2] * block_size;
    int outw = input_tensor->dims[3] * block_size;

    signed char* input_data = (signed char*)input_tensor->data;
    signed char* out_data = (signed char*)output_tensor->data;
    int total_size = input_tensor->elem_num;

    //TODO:add mode in depthtospace_param to set CRD or DCR
    for (int b = 0; b < n; ++b)
    {
        for (int s = 0; s < outc; ++s)
        {
            for (int h = 0; h < outh; ++h)
            {
                const int in_h = h / block_size;
                const int offset_h = (h % block_size);
                for (int w = 0; w < outw; ++w)
                {
                    const int in_w = w / block_size;
                    const int offset_w = w % block_size;
                    //CRD
                    const int offset_d = offset_h * block_size + offset_w;
                    const int in_d = s * (block_size * block_size) + offset_d;
                    //DCR
                    //const int offset_d =(offset_h * block_size + offset_w) * outc;
                    //const int in_d = s + offset_d;
                    const int o_index = ((b * outc + s) * outh + h) * outw + w;
                    const int i_index = ((b * inc + in_d) * inh + in_h) * inw + in_w;
                    out_data[o_index] = input_data[i_index];
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
    int layout = ir_graph->graph_layout;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct depthtospace_param* param = (struct depthtospace_param*)ir_node->op.param_mem;

    int ret = 0;
    if (input_tensor->data_type == TENGINE_DT_FP32)
        ret = ref_depthtospace_fp32(input_tensor, output_tensor, exec_graph->num_thread, param->block_size);
    else if (input_tensor->data_type == TENGINE_DT_UINT8)
        ret = ref_depthtospace_uint8(input_tensor, output_tensor, exec_graph->num_thread, param->block_size);
    else if (input_tensor->data_type == TENGINE_DT_INT8)
        ret = ref_depthtospace_int8(input_tensor, output_tensor, exec_graph->num_thread, param->block_size);

    if (ret != 0)
        return -1;

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

int register_depthtospace_ref_op()
{
    return register_builtin_node_ops(OP_DEPTHTOSPACE, &hcl_node_ops);
}

int unregister_depthtospace_ref_op()
{
    return unregister_builtin_node_ops(OP_DEPTHTOSPACE, &hcl_node_ops);
}
