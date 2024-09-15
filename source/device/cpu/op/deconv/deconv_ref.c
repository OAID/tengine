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

#include "deconv_param.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "utility/sys_port.h"
#include "utility/float.h"
#include "utility/log.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"

#include <string.h>

struct deconv_ref_param
{
    int in_shape[4];  // NCHW
    int out_shape[3]; // CHW
    int kernels[2];   // hw
    int strides[2];   // hw
    int dilations[2]; // hw
    int pads[2];
    int batch;
    int group;
    int activation;
    int layout;
    int zero[3];    // input, kernel, output
    float scale[3]; // input, kernel, output
};

static inline float activation(float input, int activation)
{
    if (activation >= 0)
    {
        if (input < 0)
            input = 0;
        if (activation == 1 && input > 1)
            input = 1;
        if (activation == 2 && input > 6)
            input = 6;
    }

    return input;
}

static int ref_deconv_fp32(const float* input, float* output, const float* kernel, const float* bias,
                           const struct deconv_ref_param* param)
{
    int batch = param->batch;
    int group = param->group;
    int input_c = param->in_shape[0] / group;
    int input_h = param->in_shape[1];
    int input_w = param->in_shape[2];
    int output_c = param->out_shape[0] / group;
    int output_h = param->out_shape[1];
    int output_w = param->out_shape[2];
    int kernel_h = param->kernels[0];
    int kernel_w = param->kernels[1];
    int pad_h0 = param->pads[0];
    int pad_w0 = param->pads[1];
    int stride_h = param->strides[0];
    int stride_w = param->strides[1];
    int dilation_h = param->dilations[0];
    int dilation_w = param->dilations[1];

    int n, g, c, h, w, kc, k_h, k_w;
    int org_out_x = 0;
    int org_out_y = 0;
    int cur_out_x = 0;
    int cur_out_y = 0;

    float input_val;
    float weight_val;
    float bias_val = 0;

    int input_offset = 0;
    int kernel_offset = 0;
    int output_offset = 0;

    memset((void*)output, 0, (unsigned long)output_h * output_w * output_c * batch * group * sizeof(float));

    for (n = 0; n < batch; ++n)
    {
        for (g = 0; g < group; ++g)
        {
            for (h = 0; h < input_h; h++)
            {
                for (w = 0; w < input_w; w++)
                {
                    org_out_x = w * stride_w - pad_w0;
                    org_out_y = h * stride_h - pad_h0;
                    for (kc = 0; kc < input_c; kc++)
                    {
                        if (param->layout == 0)
                        {
                            input_offset = n * group * input_c * input_h * input_w + g * input_c * input_h * input_w + kc * input_h * input_w + h * input_w + w;
                        }
                        else
                        {
                            input_offset = n * group * input_c * input_h * input_w + h * group * input_c * input_w + w * group * input_c + g * input_c + kc;
                        }
                        input_val = input[input_offset];
                        for (c = 0; c < output_c; c++)
                        {
                            for (k_h = 0; k_h < kernel_h; k_h++)
                            {
                                for (k_w = 0; k_w < kernel_w; k_w++)
                                {
                                    cur_out_x = org_out_x + k_w * dilation_w;
                                    cur_out_y = org_out_y + k_h * dilation_h;
                                    if (cur_out_x >= 0 && cur_out_x < output_w && cur_out_y >= 0 && cur_out_y < output_h)
                                    {
                                        if (param->layout == 0)
                                        {
                                            kernel_offset = g * output_c * input_c * kernel_h * kernel_w + kc * output_c * kernel_h * kernel_w + c * kernel_h * kernel_w + k_h * kernel_w + k_w;

                                            output_offset = n * group * output_c * output_w * output_h + g * output_c * output_w * output_h + c * output_w * output_h + cur_out_y * output_w + cur_out_x;
                                        }
                                        else
                                        {
                                            kernel_offset = g * output_c * input_c * kernel_h * kernel_w + k_h * kernel_w * output_c + k_w * output_c + c;
                                            output_offset = n * output_h * output_w * output_c * group + cur_out_y * group * output_w * output_c + cur_out_x * group * output_c + g * output_c + c;
                                        }
                                        weight_val = kernel[kernel_offset];
                                        output[output_offset] += weight_val * input_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if (NULL != bias)
    {
        for (n = 0; n < batch; n++)
        {
            for (g = 0; g < group; g++)
            {
                for (c = 0; c < output_c; c++)
                {
                    bias_val = bias[g * output_c + c];
                    for (h = 0; h < output_h; h++)
                    {
                        for (w = 0; w < output_w; w++)
                        {
                            if (param->layout == 0)
                            {
                                output_offset = n * output_c * group * output_w * output_h + g * output_c * output_w * output_h + c * output_h * output_w + h * output_w + w;
                            }
                            else
                            {
                                output_offset = n * output_c * group * output_w * output_h + h * output_c * group * output_w + w * output_c * group + c;
                            }
                            output[output_offset] += bias_val;
                        }
                    }
                }
            }
        }
    }

    // activation
    for (n = 0; n < batch * group * output_c * output_w * output_h; n++)
    {
        output[n] = activation(output[n], param->activation);
    }

    return 0;
}

static int prerun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct graph* graph = ir_node->graph;

    struct deconv_param* param = (struct deconv_param*)(ir_node->op.param_mem);
    struct deconv_ref_param* op_param = (struct deconv_ref_param*)exec_node->ops_priv;

    struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);

    if (graph->graph_layout == TENGINE_LAYOUT_NCHW) // nchw
    {
        op_param->batch = input_tensor->dims[0];
        op_param->in_shape[0] = input_tensor->dims[1];
        op_param->in_shape[1] = input_tensor->dims[2];
        op_param->in_shape[2] = input_tensor->dims[3];
    }
    else // nhwc
    {
        op_param->batch = input_tensor->dims[0];
        op_param->in_shape[0] = input_tensor->dims[3];
        op_param->in_shape[1] = input_tensor->dims[1];
        op_param->in_shape[2] = input_tensor->dims[2];
    }

    /* kernel quant param */

    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);

    if (graph->graph_layout == TENGINE_LAYOUT_NCHW) // hw
    {
        op_param->kernels[0] = weight_tensor->dims[2];
        op_param->kernels[1] = weight_tensor->dims[3];
    }
    else //
    {
        op_param->kernels[0] = weight_tensor->dims[1];
        op_param->kernels[1] = weight_tensor->dims[2];
    }

    /* output quant param */

    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (graph->graph_layout == TENGINE_LAYOUT_NCHW) // chw
    {
        op_param->out_shape[0] = output_tensor->dims[1];
        op_param->out_shape[1] = output_tensor->dims[2];
        op_param->out_shape[2] = output_tensor->dims[3];
    }
    else
    {
        op_param->out_shape[0] = output_tensor->dims[3];
        op_param->out_shape[1] = output_tensor->dims[1];
        op_param->out_shape[2] = output_tensor->dims[2];
    }

    op_param->strides[0] = param->stride_h;
    op_param->strides[1] = param->stride_w;

    op_param->dilations[1] = param->dilation_h;
    op_param->dilations[0] = param->dilation_w;

    op_param->pads[0] = param->pad_h0; // pad_h
    op_param->pads[1] = param->pad_w0; // pad_w

    op_param->group = param->group;
    op_param->activation = param->activation;
    op_param->layout = TENGINE_LAYOUT_NCHW;

    return 0;
}

static int run(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct node* ir_node = exec_node->ir_node;
    struct graph* ir_graph = ir_node->graph;
    struct tensor* i_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[0]);
    struct tensor* weight_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[1]);
    struct tensor* bias_tensor = NULL;
    struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[0]);

    if (ir_node->input_num > 2)
    {
        bias_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[2]);
    }

    void* output_data = output_tensor->data;
    const void* input_data = i_tensor->data;
    const void* kernel = weight_tensor->data;

    const void* bias = NULL;
    if (bias_tensor != NULL)
        bias = bias_tensor->data;

    struct deconv_ref_param* op_param = (struct deconv_ref_param*)exec_node->ops_priv;

    /* input quant param */
    int ret = ref_deconv_fp32((float*)input_data, (float*)output_data, (float*)kernel, (float*)bias, op_param);

    return ret;
}

static int reshape(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int postrun(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    return 0;
}

static int init_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    struct deconv_ref_param* deconv_ref_param = (struct deconv_ref_param*)sys_malloc(sizeof(struct deconv_ref_param));
    exec_node->ops_priv = deconv_ref_param;
    return 0;
}

static int release_node(struct node_ops* node_ops, struct exec_node* exec_node, struct exec_graph* exec_graph)
{
    sys_free(exec_node->ops_priv);
    exec_node->ops_priv = NULL;

    return 0;
}

static int score(struct node_ops* node_ops, struct exec_graph* exec_graph, struct node* exec_node)
{
    return OPS_SCORE_CANDO;
}

static struct node_ops hcl_node_ops = {
    .prerun = prerun,
    .run = run,
    .reshape = reshape,
    .postrun = postrun,
    .init_node = init_node,
    .release_node = release_node,
    .score = score,
};

int register_deconv_ref_op()
{
    return register_builtin_node_ops(OP_DECONV, &hcl_node_ops);
}

int unregister_deconv_ref_op()
{
    unregister_builtin_node_ops(OP_DECONV, &hcl_node_ops);
    return 0;
}
