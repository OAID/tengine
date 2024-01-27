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
 * Revised: lswang@openailab.com
 */

#include "graph/tensor.h"

#include "defines.h"
#include "api/c_api.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "utility/math.h"
#include "utility/utils.h"
#include "utility/sys_port.h"
#include "utility/log.h"

#include <stdio.h>
#include <string.h>

void init_ir_tensor(ir_tensor_t* ir_tensor, int tensor_index, int data_type)
{
    ir_tensor->index = tensor_index;
    ir_tensor->producer = -1;

    ir_tensor->consumer = (int16_t*)sys_malloc(sizeof(int16_t) * TE_MAX_CONSUMER_NUM);
    for (int i = 0; i < TE_MAX_CONSUMER_NUM; i++)
    {
        ir_tensor->consumer[i] = -1;
    }

    ir_tensor->reshaped = 0;
    ir_tensor->consumer_num = 0;
    ir_tensor->tensor_type = TENSOR_TYPE_VAR;
    ir_tensor->data_type = data_type;
    ir_tensor->dim_num = 0;
    ir_tensor->elem_size = get_tenser_element_size(data_type);
    ir_tensor->subgraph_num = 0;
    ir_tensor->free_host_mem = 0;
    ir_tensor->internal_allocated = 1;
    ir_tensor->layout = TENGINE_LAYOUT_NCHW;
    ir_tensor->quant_param_num = 0;
    ir_tensor->elem_num = 0;

    for (int i = 0; i < MAX_SHAPE_DIM_NUM; i++)
    {
        ir_tensor->dims[i] = 0;
    }

    ir_tensor->data = NULL;
    ir_tensor->name = NULL;
    ir_tensor->scale_list = NULL;
    ir_tensor->zp_list = NULL;
    ir_tensor->dev_mem = NULL;
    ir_tensor->subgraph_list = NULL;
}

ir_tensor_t* create_ir_tensor(ir_graph_t* ir_graph, const char* tensor_name, int data_type)
{
    ir_tensor_t* ir_tensor = (ir_tensor_t*)sys_malloc(sizeof(ir_tensor_t));

    if (NULL == ir_tensor)
    {
        return NULL;
    }

    init_ir_tensor(ir_tensor, ir_graph->tensor_num, data_type);

    ir_tensor->layout = ir_graph->graph_layout;

    ir_tensor_t** new_tensor_list = (ir_tensor_t**)sys_realloc(ir_graph->tensor_list, sizeof(ir_tensor_t*) * (ir_graph->tensor_num + 1));

    if (NULL == new_tensor_list)
    {
        sys_free(ir_tensor);
        return NULL;
    }

    if (NULL != tensor_name)
    {
        const int str_length = align((int)strlen(tensor_name) + 1, TE_COMMON_ALIGN_SIZE);
        ir_tensor->name = (char*)sys_malloc(str_length);

        if (NULL == ir_tensor->name)
        {
            sys_free(ir_tensor);
            return NULL;
        }

        memset(ir_tensor->name, 0, str_length);
        strcpy(ir_tensor->name, tensor_name);
    }

    new_tensor_list[ir_graph->tensor_num] = ir_tensor;

    ir_graph->tensor_list = new_tensor_list;
    ir_graph->tensor_num++;

    return ir_tensor;
}

void destroy_ir_tensor(ir_graph_t* ir_graph, ir_tensor_t* ir_tensor)
{
    if (ir_tensor->quant_param_num > 1)
    {
        sys_free(ir_tensor->scale_list);
        sys_free(ir_tensor->zp_list);
    }

    if (ir_tensor->dev_mem)
    {
        ir_node_t* ir_node = get_ir_graph_node(ir_graph, ir_tensor->producer);
        ir_subgraph_t* ir_subgraph = get_ir_graph_subgraph(ir_graph, ir_node->subgraph_idx);

        // TODO: add release impl on device
        // struct device* device = ir_subgraph->device;
        // do something to release this tensor on device

        sys_free(ir_tensor->dev_mem);
    }

    if (0 != ir_tensor->free_host_mem && NULL != ir_tensor->data)
    {
        sys_free(ir_tensor->data);
    }

    if (0 < ir_tensor->subgraph_num)
    {
        sys_free(ir_tensor->subgraph_list);
    }

    if (NULL != ir_tensor->name)
    {
        sys_free(ir_tensor->name);
    }

    if (NULL != ir_tensor->consumer)
    {
        sys_free(ir_tensor->consumer);
    }

    sys_free(ir_tensor);
}

int set_ir_tensor_shape(ir_tensor_t* tensor, const int dims[], int dim_number)
{
    if (MAX_SHAPE_DIM_NUM + 1 < dim_number)
    {
        return -1;
    }

    const int old_num = tensor->elem_num;
    int new_num = 1;

    for (int i = 0; i < dim_number; i++)
    {
        tensor->dims[i] = dims[i];
        new_num *= dims[i];
    }

    tensor->dim_num = dim_number;
    tensor->elem_num = new_num;

    if (new_num != old_num)
    {
        tensor->reshaped = tensor->consumer_num;
    }

    return 0;
}

char* create_ir_tensor_name_from_index(int index)
{
    char* name = (char*)sys_malloc(TE_COMMON_ALIGN_SIZE * 2);
    if (NULL == name)
    {
        return NULL;
    }

    sprintf(name, "tensor_%7d", index);

    return name;
}

int get_ir_tensor_index_from_name(ir_graph_t* graph, const char* tensor_name)
{
    const char* last_symbol_ptr = strrchr(tensor_name, '_');

    if (NULL != last_symbol_ptr)
    {
        const int index = atoi(++last_symbol_ptr);

        if (0 <= index && index < graph->tensor_num)
        {
            const ir_tensor_t* const tensor = graph->tensor_list[index];

            if (NULL != tensor->name && 0 == strcmp(tensor->name, tensor_name))
            {
                return index;
            }
        }
    }

    // search all tensors
    for (int i = 0; i < graph->tensor_num; i++)
    {
        const ir_tensor_t* const tensor = graph->tensor_list[i];

        if (tensor->name && 0 == strcmp(tensor->name, tensor_name))
        {
            return i;
        }
    }

    return -1;
}

int set_ir_tensor_quantization_parameter(ir_tensor_t* tensor, const float* scale, const int* zero_point, int number)
{
    if (NULL == scale || NULL == zero_point)
    {
        return -1;
    }

    if (1 == number)
    {
        tensor->scale = scale[0];
        tensor->zero_point = zero_point[0];
        tensor->quant_param_num = 1;
        return 0;
    }

    float* t_scale = (float*)sys_malloc(sizeof(float) * number);
    int* t_zero = (int*)sys_malloc(sizeof(int) * number);

    if (NULL == t_scale || NULL == t_zero)
    {
        sys_free(t_scale);
        sys_free(t_zero);
        return -1;
    }

    memcpy(t_scale, scale, sizeof(float) * number);
    memcpy(t_zero, zero_point, sizeof(int) * number);

    if (1 < tensor->quant_param_num)
    {
        sys_free(tensor->scale_list);
        sys_free(tensor->zp_list);
    }

    tensor->scale_list = t_scale;
    tensor->zp_list = t_zero;
    tensor->quant_param_num = number;

    return 0;
}

int get_ir_tensor_quantization_parameter(ir_tensor_t* tensor, float* scale, int* zero_point, int number)
{
    if (number < tensor->quant_param_num)
    {
        return -1;
    }

    if (1 == tensor->quant_param_num)
    {
        scale[0] = tensor->scale;
        zero_point[0] = tensor->zero_point;

        return 1;
    }

    memcpy(scale, tensor->scale_list, sizeof(float) * tensor->quant_param_num);
    memcpy(zero_point, tensor->zp_list, sizeof(int) * tensor->quant_param_num);

    return tensor->quant_param_num;
}

void dump_ir_tensor(ir_graph_t* g, ir_tensor_t* t)
{
    if (NULL != t->name)
    {
        TLOG_INFO("%s type: %s/%s", t->name, get_tensor_data_type_string(t->data_type), get_tensor_type_string(t->tensor_type));
    }
    else
    {
        TLOG_INFO("tensor_%d type: %s/%s", t->index, get_tensor_data_type_string(t->data_type),
                  get_tensor_type_string(t->tensor_type));
    }

    if (0 < t->dim_num)
    {
        char shape_buf[64];
        sprintf(shape_buf, " shape: [");

        for (int i = 0; i < t->dim_num - 1; i++)
        {
            sprintf(shape_buf + strlen(shape_buf), "%d,", t->dims[i]);
        }

        sprintf(shape_buf + strlen(shape_buf), "%d]", t->dims[t->dim_num - 1]);

        TLOG_INFO("%s", shape_buf);
    }
    else
    {
        TLOG_INFO(" shape: []");
    }

    if (0 <= t->producer)
    {
        ir_node_t* node = g->node_list[t->producer];

        TLOG_INFO(" from node: %d", node->index);
    }

    if (t->consumer_num > 0)
        TLOG_INFO(" (consumer: %d)", t->consumer_num);

    TLOG_INFO("\n");
}

int set_ir_tensor_consumer(ir_tensor_t* ir_tensor, const int index)
{
    if (TE_MAX_CONSUMER_NUM <= ir_tensor->consumer_num)
    {
        int16_t* new_consumer = (int16_t*)sys_realloc(ir_tensor->consumer, sizeof(int16_t) * (ir_tensor->consumer_num + 1));
        if (NULL == new_consumer)
        {
            return -1;
        }

        ir_tensor->consumer = new_consumer;
    }

    ir_tensor->consumer[ir_tensor->consumer_num] = index;
    ir_tensor->consumer_num++;

    return 0;
}

float tensor_mean(ir_tensor_t* ir_tensor)
{
    float sum = .0;
    float* p = ir_tensor->data;
    for (int i = 0; i < ir_tensor->elem_num; ++i)
    {
        sum += p[i];
    }

    float mean = sum / (float)ir_tensor->elem_num;
    return mean;
}

void save_tensor(const char* fname, const float* data, const int* dims, const int dim_num)
{
    FILE* fout = fopen(fname, "w+");
    int n = 1;
    for (int i = 0; i < dim_num; ++i)
    {
        n *= dims[i];
        fprintf(fout, "%d ", dims[i]);
    }
    fprintf(fout, "\n");

    for (int i = 0; i < n; ++i)
    {
        fprintf(fout, "%f ", data[i]);
    }
    fprintf(fout, "\n");
    fflush(fout);
    fclose(fout);
}


