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

#pragma once

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include "defines.h"

#include <stdint.h>

struct node;
struct graph;

/*!
 * @struct ir_tensor_t
 * @brief  Abstract tensor intermediate representation
 */
typedef struct tensor
{
    uint16_t index;    //!< the index of a tensor
    int16_t producer;  //!< node id, '-1' means no producer
    int16_t* consumer; //!< consumer nodes array

    uint8_t reshaped;           //!< the tensor's shape has changed
    uint8_t consumer_num;       //!< count of consumer nodes
    uint8_t tensor_type;        //!< tensor_type: { const, input, var, dep }
    uint8_t data_type;          //!< data_type: { int8, uint8, fp32, fp16, int32 }
    uint8_t dim_num;            //!< count of dimensions
    uint8_t elem_size;          //!< size of single element
    uint8_t subgraph_num;       //!< count of all subgraphs which will wait for this tensor to be ready
    uint8_t free_host_mem;      //!< should free host memory?
    uint8_t internal_allocated; //!< how memory is allocated?
    uint8_t layout;             //!< tensor layout: { TENGINE_LAYOUT_NCHW, TENGINE_LAYOUT_NHWC }

    uint16_t quant_param_num;       //!< quantization dimension
    uint32_t elem_num;              //!< count of total elements
    int dims[TE_MAX_SHAPE_DIM_NUM]; //!< shape dimensions

    /*!
     * @union anonymity data pointer
     * @brief give useful pointer pointer
     */
    union
    {
        void* data;
        int8_t* i8;
        uint8_t* u8;
        float* f32;
        uint16_t* f16;
        int32_t* i32;
    };

    char* name; //!< tensor name

    /*!
     * @union anonymity quantization scale union
     * @brief scale or its array
     */
    union
    {
        float* scale_list;
        float scale;
    };

    /*!
     * @union anonymity quantization zero point union
     * @brief zero point or its array
     */
    union
    {
        int zero_point;
        int* zp_list;
    };

    struct dev_mem* dev_mem;
    uint8_t* subgraph_list; //!< subgraph index list of those subgraphs will wait for this tensor to be ready
} ir_tensor_t;

/*!
 * @brief Create a tensor for a graph.
 *
 * @param [in]  graph: specific graph.
 * @param [in]  tensor_name: tensor name.
 * @param [in]  data_type: tensor data type(not tensor type).
 *
 * @return  The pointer of the tensor.
 */
ir_tensor_t* create_ir_tensor(struct graph* graph, const char* tensor_name, int data_type);

/*!
 * @brief Destroy a tensor.
 *
 * User should deal with other destroy works, such as ir_graph and ir_node.
 *
 * @param [in]  ir_graph: specific graph.
 * @param [in]  ir_tensor: the tensor pointer.
 */
void destroy_ir_tensor(struct graph* ir_graph, ir_tensor_t* ir_tensor);

/*!
 * @brief  Set shape for a tensor.
 *
 * @param [in]  ir_tensor: specific tensor.
 * @param [in]  dims: shape array.
 * @param [in]  dim_number: shape dimensions.
 *
 * @return statue value, 0 success, other value failure.
 */
int set_ir_tensor_shape(ir_tensor_t* ir_tensor, const int dims[], int dim_number);

/*!
 * @brief  Set tensor name from id, for anonymity ones.
 *
 * @param [in]  index: reference id.
 *
 * @return char array of the name.
 */
char* create_ir_tensor_name_from_index(int index);

/*!
 * @brief  Get tensor id from name, for anonymity ones.
 *
 * @param [in]  ir_graph: specific graph.
 * @param [in]  tensor_name: reference name.
 *
 * @return tensor id.
 */
int get_ir_tensor_index_from_name(struct graph* ir_graph, const char* tensor_name);

/*!
 * @brief  Set tensor quantization parameter.
 *
 * @param [in]  ir_tensor: specific tensor.
 * @param [in]  scale: scale pointer.
 * @param [in]  zero_point: zero_point pointer.
 * @param [in]  number: quantization parameter dimensions.
 *
 * @return statue value, 0 success, other value failure.
 */
int set_ir_tensor_quantization_parameter(ir_tensor_t* ir_tensor, const float* scale, const int* zero_point, int number);

/*!
 * @brief  Get tensor quantization parameter.
 *
 * @param [in]  ir_tensor: specific tensor.
 * @param [in]  scale: scale pointer.
 * @param [in]  zero_point: zero_point pointer.
 * @param [in]  number: quantization parameter dimensions.
 *
 * @return statue value, 0 success, other value failure.
 */
int get_ir_tensor_quantization_parameter(ir_tensor_t* ir_tensor, float* scale, int* zero_point, int number);

/*!
 * @brief  Dump the tensor.
 *
 * @param [in]  ir_graph: specific graph.
 * @param [in]  ir_tensor: specific tensor.
 */
void dump_ir_tensor(struct graph* ir_graph, ir_tensor_t* ir_tensor);

/*!
 * @brief  Set consumer node for a tensor.
 *
 * @param [in]  ir_tensor: specific tensor.
 * @param [in]  index: node index.
 *
 * @return statue value, 0 success, other value failure.
 */
int set_ir_tensor_consumer(ir_tensor_t* ir_tensor, const int index);
float tensor_mean(ir_tensor_t* tensor);

#ifdef __cplusplus
}
#endif /* __cplusplus */
