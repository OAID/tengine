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
 * Author: jiejun@openailab.com
 */

#include "cast_param.h"

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

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct cast_param* cast_param = (struct cast_param*)ir_node->op.param_mem;

    int type_from = input_tensor->data_type;
    int type_to = output_tensor->data_type;

    int num_thread = exec_graph->num_thread;

    if (input_tensor->elem_num != output_tensor->elem_num || input_tensor->dim_num != output_tensor->dim_num)
    {
        return -1;
    }

    if (type_from == type_to)
    {
        memcpy(output_tensor->data, input_tensor->data, input_tensor->elem_num * input_tensor->elem_size);
        return 0;
    }

    for (uint8_t i = 0; i < input_tensor->dim_num; i++)
    {
        if (input_tensor->dims[i] != output_tensor->dims[i])
            return -1;
    }

    if (input_tensor->layout != output_tensor->layout)
    {
        return -1;
    }

    if (type_from == TENGINE_DT_FP32 && type_to == TENGINE_DT_FP16)
    {
        fp32_t* idata = (fp32_t*)input_tensor->data;
        fp16_t* odata = (fp16_t*)output_tensor->data;

#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < input_tensor->elem_num; i++)
        {
            odata[i] = fp32_to_fp16(idata[i]);
        }

        return 0;
    }

    if (type_from == TENGINE_DT_FP16 && type_to == TENGINE_DT_FP32)
    {
        fp16_t* idata = (fp16_t*)input_tensor->data;
        fp32_t* odata = (fp32_t*)output_tensor->data;

#pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < input_tensor->elem_num; i++)
        {
            odata[i] = fp16_to_fp32(idata[i]);
        }

        return 0;
    }

    if (type_from == TENGINE_DT_FP32 && type_to == TENGINE_DT_UINT8)
    {
        float* idata = (float*)input_tensor->data;
        uint8_t* odata = (uint8_t*)output_tensor->data;

        if (1 == input_tensor->quant_param_num)
        {
            float scale = input_tensor->scale;
            int zero_point = input_tensor->zero_point;

#pragma omp parallel for num_threads(num_thread)
            for (int i = 0; i < input_tensor->elem_num; i++)
            {
                int val = (int)(roundf(idata[i] / scale)) + zero_point;

                if (255 >= val && 0 <= val)
                    odata[i] = (uint8_t)val;
                else
                {
                    if (255 < val)
                        odata[i] = 255;
                    if (0 > val)
                        odata[i] = 0;
                }
            }

            return 0;
        }
    }

    if (type_from == TENGINE_DT_UINT8 && type_to == TENGINE_DT_FP32)
    {
        uint8_t* idata = (uint8_t*)input_tensor->data;
        float* odata = (float*)output_tensor->data;

        if (1 == input_tensor->quant_param_num)
        {
            float scale = input_tensor->scale;
            int zero_point = input_tensor->zero_point;

#pragma omp parallel for num_threads(num_thread)
            for (int i = 0; i < input_tensor->elem_num; i++)
            {
                odata[i] = (float)(idata[i] - zero_point) * scale;
            }

            return 0;
        }
    }

    return -1;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* node = exec_node->ir_node;
    struct graph* ir_graph = node->graph;
    struct tensor* input = get_ir_graph_tensor(ir_graph, node->input_tensors[0]);
    struct tensor* output = get_ir_graph_tensor(ir_graph, node->output_tensors[0]);

    int ret = set_ir_tensor_shape(output, input->dims, input->dim_num);
    return ret;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    (void)node_ops;
    (void)exec_graph;
    (void)exec_node;

    return OPS_SCORE_CANDO;
}

static struct node_ops ref_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = reshape,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_cast_ref_op()
{
    return register_builtin_node_ops(OP_CAST, &ref_node_ops);
}

int unregister_cast_ref_op()
{
    return unregister_builtin_node_ops(OP_CAST, &ref_node_ops);
}
