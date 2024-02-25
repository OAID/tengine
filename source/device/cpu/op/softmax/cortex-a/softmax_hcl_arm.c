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
 * Author: haoluo@openailab.com
 */

#include "softmax_param.h"

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

#include <arm_neon.h>

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor;
    struct tensor* output_tensor;
    int ret = 0;

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (input_tensor->dims[0] != output_tensor->dims[0] || input_tensor->dims[1] != output_tensor->dims[1] || input_tensor->dims[2] != output_tensor->dims[2] || input_tensor->dims[3] != output_tensor->dims[3])
        ret = set_ir_tensor_shape(output_tensor, input_tensor->dims, input_tensor->dim_num);

    return ret;
}

static inline float32x4_t vexpq10_f32(float32x4_t x)
{
    x = vmlaq_n_f32(vdupq_n_f32(1.0f), x, 0.0009765625f); // n = 10
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    x = vmulq_f32(x, x);
    return x;
}

static void GetMaxArray(float* input, float* array, int in_size, int on_size, int num_thread)
{
    float* input_ptr = (float*)input;
    float* array_ptr = (float*)array;
    memset(array, 0, in_size * sizeof(float));

    // #pragma omp parallel for num_threads(num_thread)
    for (int j = 0; j < on_size; j++)
    {
        // #pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < (in_size & -4); i += 4)
        {
            float32x4_t _p = vld1q_f32(array_ptr + i);
            float32x4_t _in = vld1q_f32(input_ptr + j * in_size + i);
#ifdef __aarch64__
            _p = vpmaxq_f32(_p, _in);
#else
            _p = vmaxq_f32(_p, vrev64q_f32(_in));
            _p = vmaxq_f32(_p, vextq_f32(_p, _in, 2));
#endif
            vst1q_f32(array_ptr + i, _p);
        }
        for (int i = in_size & ~3; i < in_size; i++)
        {
            if (array_ptr[i] < input_ptr[j * in_size + i])
                array_ptr[i] = input_ptr[j * in_size + i];
        }
        /*
        for(int l = 0; l < in_size; l++)
        {
            if(array_ptr[l] < input_ptr[j * in_size + l])
                array_ptr[l] = input_ptr[j * in_size + l];
        }
        */
    }
}

static void GetOutResult(float* input, float* output, float* maxarray, float* sum_array, int in_size, int on_size,
                         int num_thread)
{
    float* input_ptr = (float*)input;
    float* output_ptr = (float*)output;
    float* maxarray_ptr = (float*)maxarray;
    float* sum_array_ptr = (float*)sum_array;

    memset(sum_array, 0x0, in_size * sizeof(float));

    /* get the exp and the summary */
    // #pragma omp parallel for num_threads(num_thread)
    for (int j = 0; j < on_size; j++)
    {
        // #pragma omp parallel for num_threads(num_thread)
        for (int i = 0; i < (in_size & -4); i += 4)
        {
            int index = j * in_size + i;
            float32x4_t out = vexpq10_f32(vsubq_f32(vld1q_f32(input_ptr + index), vld1q_f32(maxarray_ptr + i)));
            float32x4_t sum = vaddq_f32(vld1q_f32(sum_array_ptr + i), out);
            vst1q_f32(output_ptr + index, out);
            vst1q_f32(sum_array_ptr + i, sum);
        }
        for (int i = in_size & ~3; i < in_size; i++)
        {
            int index = j * in_size + i;
            output_ptr[index] = exp(input_ptr[index] - maxarray_ptr[i]);
            sum_array_ptr[i] += output_ptr[index];
        }
    }
    /*
        for(int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] = exp(input_ptr[index] - array_ptr[l]);
            sum_array_ptr[l] += output_ptr[index];
        }
    */
    /* the final result */
    for (int j = 0; j < on_size; j++)
        for (int l = 0; l < in_size; l++)
        {
            int index = j * in_size + l;
            output_ptr[index] /= sum_array_ptr[l];
        }
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

    input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);
    struct softmax_param* softmax_param = (struct softmax_param*)ir_node->op.param_mem;

    int element_size = input_tensor->elem_size;
    int dims[4];

    for (int i = 0; i < input_tensor->dim_num; i++)
    {
        dims[i] = input_tensor->dims[i];
    }

    int axis = softmax_param->axis;
    int out_size, in_size, on_size;

    out_size = 1;
    for (int i = 0; i < axis; i++)
    {
        out_size *= dims[i];
    }

    in_size = 1;
    for (size_t i = axis + 1; i < input_tensor->dim_num; i++)
    {
        in_size *= dims[i];
    }
    on_size = dims[axis];

    uint8_t* input = input_tensor->data;
    uint8_t* output = output_tensor->data;
    float* max_array = (float*)malloc(in_size * sizeof(float));
    float* sum_array = (float*)malloc(in_size * sizeof(float));

    int on_in_size = on_size * in_size;

    float* input_f = NULL;
    float* output_f = NULL;

    if (element_size == 1)
    {
        input_f = (float*)malloc(on_in_size * 4);
        output_f = (float*)malloc(on_in_size * 4);

        /* todo */

        free(input_f);
        free(output_f);
    }

    for (int i = 0; i < out_size; i++)
    {
        /* get max */
        int img_base = i * on_in_size * element_size;

        GetMaxArray((float*)(input + img_base), max_array, in_size, on_size, exec_graph->num_thread);
        GetOutResult((float*)(input + img_base), (float*)(output + img_base), max_array, sum_array, in_size,
                     on_size, exec_graph->num_thread);
    }

    free(max_array);
    free(sum_array);

    return 0;
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

int register_softmax_hcl_arm_op()
{
    return register_builtin_node_ops(OP_SOFTMAX, &hcl_node_ops);
}

int unregister_softmax_hcl_arm_op()
{
    return unregister_builtin_node_ops(OP_SOFTMAX, &hcl_node_ops);
}
