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
 * Author: sqfu@openailab.com
 * Author: qtang@openailab.com (update 20200611)
 */

#include "pad_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"
#include "memory.h"
#include <math.h>

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static void ref_pad_fp32(float* input, float* output, int in_h, int in_w, int out_h, int out_w, int top, int left, float v, int mode)
{
    float* ptr = input;
    float* outptr = output;
    int y = 0;
    // fill top

    if (0 == mode)
    {
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < out_w; x++)
            {
                outptr[x] = v;
            }
            outptr += out_w;
        }
        // fill center
        for (; y < (top + in_h); y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = v;
            }
            for (x = 0; x < in_w; x++)
            {
                outptr[left + x] = ptr[x];
            }
            for (x = out_w - left; x < out_w; x++)
            {
                outptr[x] = v;
            }
            ptr += in_w;
            outptr += out_w;
        }
        // fill bottom
        for (; y < out_h; y++)
        {
            int x = 0;
            for (; x < out_w; x++)
            {
                outptr[x] = v;
            }
            outptr += out_w;
        }
    }
    else if (1 == mode)
    {
        for (; y < top; y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }
            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - 1];
            }

            outptr += out_w;
        }
        // fill center
        for (; y < (top + in_h); y++)
        {
            int x = 0;

            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x += in_w;
            }

            // paddding right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - 1];
            }
            ptr += in_w;
            outptr += out_w;
        }
        // fill bottom
        for (; y < out_h; y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }
            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - 1];
            }
            outptr += out_w;
        }
    }
    else if (2 == mode)
    {
        ptr += top * in_w;
        for (; y < top; y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }

            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - (x - in_w - left) - 2];
            }
            outptr += out_w;
            ptr -= in_w;
        }

        // fill center
        for (; y < (top + in_h); y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }

            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - (x - in_w - left) - 2];
            }
            outptr += out_w;
            ptr += in_w;
        }
        ptr -= 2 * in_w;
        // fill bottom
        for (; y < out_h; y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }

            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - (x - in_w - left) - 2];
            }
            outptr += out_w;
            ptr -= in_w;
        }
    }
}

static void ref_pad_uint8(uint8_t* input, uint8_t* output, int in_h, int in_w, int out_h, int out_w, int top, int left, float v, int mode)
{
    uint8_t* ptr = input;
    uint8_t* outptr = output;

    int y = 0;
    // fill top

    if (0 == mode)
    {
        for (; y < top; y++)
        {
            int x = 0;
            for (; x < out_w; x++)
            {
                outptr[x] = v;
            }
            outptr += out_w;
        }
        // fill center
        for (; y < (top + in_h); y++)
        {
            int x = 0;
            for (; x < left; x++)
            {
                outptr[x] = v;
            }
            for (x = 0; x < in_w; x++)
            {
                outptr[left + x] = ptr[x];
            }
            for (x = out_w - left; x < out_w; x++)
            {
                outptr[x] = v;
            }
            ptr += in_w;
            outptr += out_w;
        }
        // fill bottom
        for (; y < out_h; y++)
        {
            int x = 0;
            for (; x < out_w; x++)
            {
                outptr[x] = v;
            }
            outptr += out_w;
        }
    }
    else if (1 == mode)
    {
        for (; y < top; y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }
            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - 1];
            }

            outptr += out_w;
        }
        // fill center
        for (; y < (top + in_h); y++)
        {
            int x = 0;

            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x += in_w;
            }

            // paddding right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - 1];
            }
            ptr += in_w;
            outptr += out_w;
        }
        // fill bottom
        for (; y < out_h; y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[0];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }
            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - 1];
            }
            outptr += out_w;
        }
    }
    else if (2 == mode)
    {
        ptr += top * in_w;
        for (; y < top; y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }

            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - (x - in_w - left) - 2];
            }
            outptr += out_w;
            ptr -= in_w;
        }

        // fill center
        for (; y < (top + in_h); y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }

            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - (x - in_w - left) - 2];
            }
            outptr += out_w;
            ptr += in_w;
        }
        ptr -= 2 * in_w;
        // fill bottom
        for (; y < out_h; y++)
        {
            int x = 0;
            // padding left
            for (; x < left; x++)
            {
                outptr[x] = ptr[left - x];
            }

            // padding center
            if (in_w < 12)
            {
                for (; x < (left + in_w); x++)
                {
                    outptr[x] = ptr[x - left];
                }
            }
            else
            {
                memcpy(outptr + left, ptr, sizeof(float) * in_w);
                x = x + in_w;
            }

            //pading right
            for (; x < out_w; x++)
            {
                outptr[x] = ptr[in_w - (x - in_w - left) - 2];
            }
            outptr += out_w;
            ptr -= in_w;
        }
    }
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    struct pad_param* param = (struct pad_param*)ir_node->op.param_mem;

    int batch = input_tensor->dims[0];
    int in_c = input_tensor->dims[1];
    int in_h = input_tensor->dims[2];
    int in_w = input_tensor->dims[3];
    int in_cstep = in_h * in_w;
    int in_size = in_c * in_h * in_w;

    int out_h = output_tensor->dims[2];
    int out_w = output_tensor->dims[3];
    int out_c = output_tensor->dims[1];

    int out_cstep = out_h * out_w;
    int out_size = out_c * out_h * out_w;

    int pad_top = param->pad_2_h;
    int pad_bottom = param->pad_2_w;
    int pad_left = param->pad_3_h;
    int pad_right = param->pad_3_w;
    int pad_front = param->pad_1_h;
    int pad_back = param->pad_1_w;

    if ((param->mode != 0) && (param->mode != 1) && (param->mode != 2))
    {
        TLOG_ERR("another mode dose not support, pad mode value %d\n", param->mode);
        return -1;
    }

    for (int n = 0; n < batch; n++)
    {
        // padding wh
        for (int c = 0; c < in_c; c++)
        {
            if (input_tensor->data_type == TENGINE_DT_FP32)
            {
                float* input_data = (float*)input_tensor->data + n * in_size + c * in_cstep;
                float* output_data = (float*)output_tensor->data + n * out_size + (c + pad_front) * out_cstep;
                ref_pad_fp32(input_data, output_data, in_h, in_w, out_h, out_w, pad_top, pad_left, param->value, param->mode);
            }
            else if (input_tensor->data_type == TENGINE_DT_UINT8)
            {
                uint8_t* input_data = (uint8_t*)input_tensor->data + n * in_size + c * in_cstep;
                uint8_t* output_data = (uint8_t*)output_tensor->data + n * out_size + (c + pad_front) * out_cstep;
                ref_pad_uint8(input_data, output_data, in_h, in_w, out_h, out_w, pad_top, pad_left, param->value, param->mode);
            }
        }

        // padding channel : front
        for (int c = 0; c < pad_front; c++)
        {
            if (0 == param->mode)
            {
                if (input_tensor->data_type == TENGINE_DT_FP32)
                {
                    float* output_data = (float*)output_tensor->data + n * out_size + c * out_cstep;
                    memset(output_data, param->value, sizeof(float) * out_cstep);
                }
                else if (input_tensor->data_type == TENGINE_DT_UINT8)
                {
                    uint8_t* output_data = (uint8_t*)output_tensor->data + n * out_size + c * out_cstep;
                    memset(output_data, param->value, sizeof(uint8_t) * out_cstep);
                }
            }
            else
            {
                if (input_tensor->data_type == TENGINE_DT_FP32)
                {
                    float* output_data = (float*)output_tensor->data + n * out_size + c * out_cstep;
                    float* copy_data = (float*)output_tensor->data + n * out_size + pad_front * out_cstep;
                    memcpy(output_data, copy_data, sizeof(float) * out_cstep);
                }
                else if (input_tensor->data_type == TENGINE_DT_UINT8)
                {
                    uint8_t* output_data = (uint8_t*)output_tensor->data + n * out_size + c * out_cstep;
                    uint8_t* copy_data = (uint8_t*)output_tensor->data + n * out_size + pad_front * out_cstep;
                    memcpy(output_data, copy_data, sizeof(uint8_t) * out_cstep);
                }
            }
        }

        // padding channel : back
        for (int c = pad_front + in_c; c < out_c; c++)
        {
            if (0 == param->mode)
            {
                if (input_tensor->data_type == TENGINE_DT_FP32)
                {
                    float* output_data = (float*)output_tensor->data + n * out_size + c * out_cstep;
                    memset(output_data, param->value, sizeof(float) * out_cstep);
                }
                else if (input_tensor->data_type == TENGINE_DT_UINT8)
                {
                    uint8_t* output_data = (uint8_t*)output_tensor->data + n * out_size + c * out_cstep;
                    memset(output_data, param->value, sizeof(uint8_t) * out_cstep);
                }
            }
            else
            {
                if (input_tensor->data_type == TENGINE_DT_FP32)
                {
                    float* output_data = (float*)output_tensor->data + n * out_size + c * out_cstep;
                    float* copy_data = (float*)output_tensor->data + n * out_size + (pad_front + in_c) * out_cstep;
                    memcpy(output_data, copy_data, sizeof(float) * out_cstep);
                }
                else if (input_tensor->data_type == TENGINE_DT_UINT8)
                {
                    uint8_t* output_data = (uint8_t*)output_tensor->data + n * out_size + c * out_cstep;
                    uint8_t* copy_data = (uint8_t*)output_tensor->data + n * out_size + (pad_front + in_c) * out_cstep;
                    memcpy(output_data, copy_data, sizeof(uint8_t) * out_cstep);
                }
            }
        }
    }

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_BEST;
}

static struct node_ops pad_node_ops = {
    .prerun = NULL,
    .run = run,
    .reshape = NULL,
    .postrun = NULL,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_pad_ref_op()
{
    return register_builtin_node_ops(OP_PAD, &pad_node_ops);
}

int unregister_pad_ref_op()
{
    return unregister_builtin_node_ops(OP_PAD, &pad_node_ops);
}
