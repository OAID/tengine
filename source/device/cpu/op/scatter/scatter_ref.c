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

#include "scatter_param.h"

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
#include <stdbool.h>
#include <string.h>

struct ref_scatter_param
{
    int axis;
    bool is_onnx;
    int* update_dim;
    int* indice_dim;
    int dims[4];
    int dim_size;
    int updateSize;
    int indiceSize;
};

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ref_scatter_param* scatter_op_param = (struct ref_scatter_param*)sys_malloc(sizeof(struct ref_scatter_param));
    memset(scatter_op_param, 0, sizeof(struct ref_scatter_param));
    exec_node->ops_priv = scatter_op_param;
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
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ref_scatter_param* scatter_op_param = (struct ref_scatter_param*)exec_node->ops_priv;
    struct scatter_param* param = (struct scatter_param*)(ir_node->op.param_mem);
    scatter_op_param->dim_size = input_tensor->dim_num;
    scatter_op_param->is_onnx = param->is_onnx;
    for (int i = 0; i < 4; i++)
    {
        scatter_op_param->dims[i] = 1;
    }

    if (scatter_op_param->is_onnx)
    {
        struct tensor* indices_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
        int indicesDimsSize = indices_tensor->dim_num;
        scatter_op_param->indice_dim = (int*)malloc(sizeof(int) * indicesDimsSize);
        scatter_op_param->indiceSize = indicesDimsSize;

        struct tensor* updates_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        int updatesDimsSize = updates_tensor->dim_num;
        scatter_op_param->update_dim = (int*)malloc(sizeof(int) * updatesDimsSize);
        scatter_op_param->updateSize = updatesDimsSize;
    }

    return 0;
}

static int ref_scatter_fp32(float* input, float* output, int* indices, float* updates, struct ref_scatter_param* op_param)
{
    int axis = op_param->axis;
    bool is_onnx = op_param->is_onnx;
    TLOG_ERR("updates %f %f \n", updates[0], updates[1]);
    TLOG_ERR("indices %d %d \n", indices[0], indices[1]);
    int outSize = 1;
    for (int i = 0; i < op_param->dim_size; i++)
    {
        outSize *= op_param->dims[4 - op_param->dim_size + i];
    }
    memcpy(output, input, sizeof(float) * outSize);

    int calIndexDims[4];
    int realIndexDims[4];
    int outCalAxis[4];
    int outRealAxis[4];
    int updateDims[4];

    for (int i = 0; i < 4; i++)
    {
        calIndexDims[i] = 0;
        realIndexDims[i] = 1;
        outCalAxis[i] = 0;
        outRealAxis[i] = 0;
        updateDims[i] = 1;
    }

    int diff = 4 - op_param->updateSize;
    //TLOG_ERR("update size: %d \n", op_param->updateSize);
    for (int i = 0; i < op_param->updateSize; i++)
    {
        calIndexDims[diff + i] = op_param->update_dim[i];
        realIndexDims[diff + i] = op_param->update_dim[i];
        TLOG_ERR("%d %d \n", calIndexDims[diff + i], realIndexDims[diff + i]);
    }

    diff = 4 - op_param->dim_size;
    for (int i = 0; i < op_param->dim_size; i++)
    {
        outCalAxis[diff + i] = 1;
        outRealAxis[diff + i] = op_param->dims[diff + i];
    }
    outCalAxis[diff + op_param->axis] = 2;

    int inN = op_param->dims[0];
    int inC = op_param->dims[1];
    int inH = op_param->dims[2];
    int inW = op_param->dims[3];
    TLOG_ERR("Ready for test\n");
    // TLOG_ERR("reaslIndexDims: %d %d %d %d \n", realIndexDims[0] ,realIndexDims[1], realIndexDims[2],realIndexDims[3]);
    // op_param->axis = -1;
    if (is_onnx)
    {
        if (op_param->axis != -1)
        {
            if (op_param->dim_size == 1)
            {
                TLOG_ERR("dims 1\n");
                for (int n = 0; n < realIndexDims[0]; n++)
                {
                    for (int c = 0; c < realIndexDims[1]; c++)
                    {
                        for (int h = 0; h < realIndexDims[2]; h++)
                        {
                            for (int w = 0; w < realIndexDims[3]; w++)
                            {
                                int ii = n * calIndexDims[1] * calIndexDims[2] * calIndexDims[3] + c * calIndexDims[2] * calIndexDims[3] + h * calIndexDims[3] + w;
                                int index = indices[ii];
                                if (index < 0)
                                {
                                    index = inW + index + 1;
                                }
                                float value = updates[ii];

                                int outIndex = index;
                                output[outIndex] = value;
                            }
                        }
                    }
                }
            }
            else if (op_param->dim_size == 2)
            {
                TLOG_ERR("dims 2 in \n");
                for (int n = 0; n < realIndexDims[0]; n++)
                {
                    for (int c = 0; c < realIndexDims[1]; c++)
                    {
                        for (int h = 0; h < realIndexDims[2]; h++)
                        {
                            for (int w = 0; w < realIndexDims[3]; w++)
                            {
                                TLOG_ERR("cadsfasd \n");
                                int ii = n * calIndexDims[1] * calIndexDims[2] * calIndexDims[3] + c * calIndexDims[2] * calIndexDims[3] + h * calIndexDims[3] + w;
                                TLOG_ERR("cadsfasd 2 %d \n", ii);
                                int index = indices[ii];
                                TLOG_ERR("cadsfasd 3\n");
                                float value = updates[ii];
                                TLOG_ERR("dims 2ddd\n");
                                if (op_param->axis == 1)
                                {
                                    index = index < 0 ? inW + index + 1 : index;

                                    int outIndex = h * realIndexDims[3] + index;
                                    TLOG_ERR("%d %d \n", index, outIndex);
                                    output[outIndex] = value;
                                }
                                if (op_param->axis == 0)
                                {
                                    index = index < 0 ? inH + index + 1 : index;

                                    int outIndex = index * realIndexDims[3] + w;
                                    TLOG_ERR("%d %d \n", index, outIndex);
                                    output[outIndex] = value;
                                }
                            }
                        }
                    }
                }
            }
            else if (op_param->dim_size == 3)
            {
                TLOG_ERR("dims 3\n");
                for (int n = 0; n < realIndexDims[0]; n++)
                {
                    for (int c = 0; c < realIndexDims[1]; c++)
                    {
                        for (int h = 0; h < realIndexDims[2]; h++)
                        {
                            for (int w = 0; w < realIndexDims[3]; w++)
                            {
                                int ii = n * calIndexDims[1] * calIndexDims[2] * calIndexDims[3] + c * calIndexDims[2] * calIndexDims[3] + h * calIndexDims[3] + w;
                                int index = indices[ii];
                                float value = updates[ii];

                                if (op_param->axis == 1)
                                {
                                    index = index < 0 ? inH + index + 1 : index;
                                    int outIndex = c * inH * inW + index * realIndexDims[3] + w;
                                    output[outIndex] = value;
                                }
                                if (op_param->axis == 0)
                                {
                                    index = index < 0 ? inC + index + 1 : index;
                                    // TLOG_ERR("%d \n", index);
                                    int outIndex = index * inH * inW + h * realIndexDims[3] + w;
                                    output[outIndex] = value;
                                }
                                if (op_param->axis == 2)
                                {
                                    index = index < 0 ? inW + index + 1 : index;
                                    int outIndex = c * inH * inW + h * realIndexDims[3] + index;
                                    output[outIndex] = value;
                                }
                            }
                        }
                    }
                }
            }
            else if (op_param->dim_size == 4)
            {
                TLOG_ERR("dims 4\n");
                for (int n = 0; n < realIndexDims[0]; n++)
                {
                    for (int c = 0; c < realIndexDims[1]; c++)
                    {
                        for (int h = 0; h < realIndexDims[2]; h++)
                        {
                            for (int w = 0; w < realIndexDims[3]; w++)
                            {
                                int ii = n * calIndexDims[1] * calIndexDims[2] * calIndexDims[3] + c * calIndexDims[2] * calIndexDims[3] + h * calIndexDims[3] + w;
                                int index = indices[ii];
                                float value = updates[ii];

                                if (op_param->axis == 1)
                                {
                                    index = index < 0 ? inC + index + 1 : index;
                                    int outIndex = n * inC * inH * inW + index * inH * inW + h * realIndexDims[3] + w;
                                    output[outIndex] = value;
                                }
                                if (op_param->axis == 0)
                                {
                                    index = index < 0 ? inN + index + 1 : index;
                                    int outIndex = index * inC * inH * inW + c * inH * inW + h * realIndexDims[3] + w;
                                    output[outIndex] = value;
                                }
                                if (op_param->axis == 2)
                                {
                                    index = index < 0 ? inH + index + 1 : index;
                                    int outIndex = n * inC * inH * inW + c * inH * inW + index * realIndexDims[3] + w;
                                    output[outIndex] = value;
                                }
                                if (op_param->axis == 3)
                                {
                                    index = index < 0 ? inW + index + 1 : index;
                                    int outIndex = n * inC * inH * inW + c * inH * inW + h * realIndexDims[3] + index;
                                    output[outIndex] = value;
                                }
                            }
                        }
                    }
                }
            }
        }
        else
        {
            int data_dims[4] = {1};
            for (int i = 0; i < op_param->dim_size; i++)
            {
                data_dims[3 - i] = op_param->dims[i];
            }

            int iCHW = data_dims[1] * data_dims[2] * data_dims[3];
            int iHW = data_dims[2] * data_dims[3];

            for (int i = 0; i < op_param->updateSize; i++)
            {
                updateDims[4 - op_param->updateSize + i] = op_param->update_dim[i];
            }

            int uCHW = updateDims[1] * updateDims[2] * updateDims[3];
            int uHW = updateDims[2] * updateDims[3];
            for (int n = 0; n < updateDims[0]; n++)
            {
                for (int c = 0; c < updateDims[1]; c++)
                {
                    for (int h = 0; h < updateDims[2]; h++)
                    {
                        for (int w = 0; w < updateDims[3]; w++)
                        {
                            int updateIndex = n * uCHW + c * uHW + h * updateDims[3] + w;
                            int value = updates[updateIndex];
                            int index = indices[updateIndex];
                            int outIndex = n * iCHW + c * iHW + w * updateDims[2] + index;
                            output[outIndex] = value;
                        }
                    }
                }
            }
        }
    }
    else
    {
        return -1;
    }

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct ref_scatter_param* scatter_op_param = (struct ref_scatter_param*)exec_node->ops_priv;
    struct scatter_param* param = (struct scatter_param*)(ir_node->op.param_mem);

    int inputDimsSize = input_tensor->dim_num;
    for (int i = 0; i < inputDimsSize; i++)
    {
        scatter_op_param->dims[4 - inputDimsSize + i] = input_tensor->dims[i];
    }
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    scatter_op_param->axis = param->axis;
    scatter_op_param->is_onnx = param->is_onnx;
    if (scatter_op_param->is_onnx)
    {
        struct tensor* indices_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
        int indicesDimsSize = indices_tensor->dim_num;
        for (int i = 0; i < indicesDimsSize; i++)
        {
            scatter_op_param->indice_dim[i] = indices_tensor->dims[i];
        }
        struct tensor* updates_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
        int updatesDimsSize = updates_tensor->dim_num;
        for (int i = 0; i < updatesDimsSize; i++)
        {
            scatter_op_param->update_dim[i] = updates_tensor->dims[i];
        }
        TLOG_ERR("Indecues %d \n", indicesDimsSize);

        int ret = ref_scatter_fp32((float*)input_tensor->data, (float*)output_tensor->data,
                                   (int*)indices_tensor->data, (float*)updates_tensor->data, scatter_op_param);
        if (ret < 0)
        {
            TLOG_ERR("Scatter reference error \n");
        }
    }
    else
    {
        return -1;
    }

    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct ref_scatter_param* scatter_op_param = (struct ref_scatter_param*)exec_node->ops_priv;

    sys_free(scatter_op_param->indice_dim);
    sys_free(scatter_op_param->update_dim);

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_scatter_ref_op()
{
    return register_builtin_node_ops(OP_SCATTER, &hcl_node_ops);
}

int unregister_scatter_ref_op()
{
    return unregister_builtin_node_ops(OP_SCATTER, &hcl_node_ops);
}
