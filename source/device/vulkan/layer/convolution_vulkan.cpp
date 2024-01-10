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
 * Parts of the following code in this file refs to
 * https://github.com/Tencent/ncnn/tree/master/src/layer/vulkan/
 * Tencent is pleased to support the open source community by making ncnn
 * available.
 *
 * Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
 *
 * Licensed under the BSD 3-Clause License (the "License"); you may not use this
 * file except in compliance with the License. You may obtain a copy of the
 * License at
 *
 * https://opensource.org/licenses/BSD-3-Clause
 */

/*
 * Copyright (c) 2020, Open AI Lab
 * Author: ddzhao@openailab.com
 */

#include "convolution_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

Convolution_vulkan::Convolution_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = true;
    padding = 0;
    innerproduct = 0;

    pipeline_convolution = 0;
    pipeline_convolution_pack4 = 0;
    pipeline_convolution_pack8 = 0;
    pipeline_convolution_pack1to4 = 0;
    pipeline_convolution_pack4to1 = 0;
    pipeline_convolution_pack1to8 = 0;
    pipeline_convolution_pack4to8 = 0;
    pipeline_convolution_pack8to1 = 0;
    pipeline_convolution_pack8to4 = 0;
    pipeline_convolution_1x1s1d1 = 0;
    pipeline_convolution_pack4_1x1s1d1 = 0;
    pipeline_convolution_pack8_1x1s1d1 = 0;

    graph = ir_graph;
    node = ir_node;

    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name = input->name;
    bottoms.push_back(name);

    // Tensor* output_tensor = t_node->GetOutputTensor(0);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    name = output->name;
    tops.push_back(name);

    // Convolution* conv_op = dynamic_cast<Convolution*>(node->GetOp());
    // ConvParam* param = conv_op->GetParam();
    struct conv_param* param = (struct conv_param*)ir_node->op.param_mem;

    group = param->group;
    input_c = input->dims[1]; // param->input_channel;
    input_h = input->dims[2];
    input_w = input->dims[3];
    pad_w0 = param->pad_w0; // left padding columns
    pad_w1 = param->pad_w1; // right padding columns
    pad_h0 = param->pad_h0; // top padding rows
    pad_h1 = param->pad_h1; // bottom padding rows
    stride_w = param->stride_w;
    stride_h = param->stride_h;
    dilation_w = param->dilation_w;
    dilation_h = param->dilation_h;
    kernel_w = param->kernel_w;
    kernel_h = param->kernel_h;
    activation = param->activation == 0 ? 1 : -1;
    output_c = output->dims[1]; // param->output_channel;
    output_h = output->dims[2];
    output_w = output->dims[3];
    struct tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    weight_data_size = weight->elem_num;
}

int Convolution_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    // const Tshape& shape = bottom_shapes.empty() ? Tshape() : bottom_shapes[0];
    // const Tshape& out_shape = top_shapes.empty() ? Tshape() : top_shapes[0];

    // const int maxk = kernel_w * kernel_h;
    // // int num_input = weight_data_size / maxk / num_output;
    // int num_output = output_c;
    // int num_input = input_c;
    const Tshape& shape = Tshape(input_w, input_h, input_c);
    const Tshape& out_shape = Tshape(output_w, output_h, output_c);
    const int maxk = kernel_w * kernel_h;
    int num_output = output_c;
    int num_input = input_c;
    int pad_left = pad_w0;
    int pad_right = pad_w1;
    int pad_top = pad_h0;
    int pad_bottom = pad_h1;

    // TLOG_INFO("%d %d %d -> %d %d %d\n", shape.c, shape.h, shape.w, out_shape.c, out_shape.h, out_shape.w);
    // fc
    // if (kernel_w == 1 && kernel_h == 1)
    // {
    //     innerproduct = new InnerProduct_vulkan(graph, node);
    //     innerproduct->vkdev = vkdev;

    //     innerproduct->create_pipeline(opt);

    //     if (shape.dims == 1 && shape.w == num_input)
    //     {
    //         return 0;
    //     }
    // }

    Tshape shape_bordered = Tshape();

    if (shape.dims != 0)
    {
        if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
        {
            shape_bordered = Tshape(shape.w + pad_left + pad_right, shape.h + pad_top + pad_bottom, shape.c);
        }
        else if ((pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
                 || (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234))
        {
            const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
            const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

            int wpad = kernel_extent_w + (shape.w - 1) / stride_w * stride_w - shape.w;
            int hpad = kernel_extent_h + (shape.h - 1) / stride_h * stride_h - shape.h;
            if (wpad > 0 || hpad > 0)
            {
                shape_bordered = Tshape(shape.w + wpad, shape.h + hpad, shape.c);
            }
        }
        else
        {
            shape_bordered = shape;
        }
    }

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4
                                                                                       : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4
                                                                                             : 1;

    size_t elemsize;
    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
        out_elemsize = out_elempack * 4u;
    }

    // TLOG_INFO("elemsize out_elemsize:%d %d\n", elemsize, out_elemsize);

    Tshape shape_bordered_packed;
    // if (shape_bordered.dims == 3) shape_bordered_packed = Mat(shape_bordered.w, shape_bordered.h, num_input / elempack, (void*)0, elemsize, elempack);
    if (shape_bordered.dims == 3) shape_bordered_packed = Tshape(shape_bordered.w, shape_bordered.h, num_input / elempack);

    Tshape out_shape_packed;
    // if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, num_output / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Tshape(out_shape.w, out_shape.h, num_output / out_elempack);

    bool is_conv1x1s1d1 = kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    // bool is_conv3x3s1d1 = kernel_w == 3 && kernel_h == 3 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1;
    // bool is_conv1x1s1d1 = false;
    bool is_conv3x3s1d1 = false;

    {
        opt.use_image_storage = false;
    }

    {
        padding = new Padding_vulkan(vkdev);
        padding->vkdev = vkdev;

        padding->top = pad_h0;
        padding->bottom = pad_h1;
        padding->left = pad_w0;
        padding->right = pad_w1;
        padding->type = 0;
        padding->value = 0;

        padding->input_w = input_w;
        padding->input_h = input_h;
        padding->input_c = input_c;
        padding->output_w = input_w + pad_w0 + pad_w1;
        padding->output_h = input_h + pad_h0 + pad_h1;
        padding->output_c = input_c;

        padding->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(10 + 10);
    specializations[0].i = kernel_w;                    // kernel_w;
    specializations[1].i = kernel_h;                    // kernel_h
    specializations[2].i = dilation_w;                  // dilation_w;
    specializations[3].i = dilation_h;                  // dilation_h;
    specializations[4].i = stride_w;                    // stride_w;
    specializations[5].i = stride_h;                    // stride_h;
    specializations[6].i = node->input_num > 2 ? 1 : 0; // bias_term;
    specializations[7].i = activation;                  // activation_type;
    specializations[8].f = 0;                           //param->activation;	// activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[9].f = 0;                           //param->activation; 	// activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[10 + 0].i = 0;                      //3;	// shape_bordered_packed.dims;
    specializations[10 + 1].i = 0;                      //input_w + pad_w0 + pad_w1;	// shape_bordered_packed.w;
    specializations[10 + 2].i = 0;                      //input_h + pad_h0 + pad_h1;	// shape_bordered_packed.h;
    specializations[10 + 3].i = 0;                      //input_c;	// shape_bordered_packed.c;
    specializations[10 + 4].i = 0;                      //(input_w + pad_w0 + pad_w1) * (input_h + pad_h0 + pad_h1);	// shape_bordered_packed.cstep;
    specializations[10 + 5].i = 0;                      // out_shape_packed.dims;
    specializations[10 + 6].i = 0;                      //output_w;	// out_shape_packed.w;
    specializations[10 + 7].i = 0;                      //output_h;	// out_shape_packed.h;
    specializations[10 + 8].i = 0;                      //output_c;	// out_shape_packed.c;
    specializations[10 + 9].i = 0;                      //output_w * output_h;	// out_shape_packed.cstep;

    // TODO with local_size_xyz and shader_index options

    VkTensor local_size_xyz;
    local_size_xyz.w = std::min(8, out_shape_packed.w);
    local_size_xyz.h = std::min(8, out_shape_packed.h);
    local_size_xyz.c = std::min(4, out_shape_packed.c);

    // TLOG_INFO("create pipeline elempack out_elempack:%d %d\n", elempack, out_elempack);

    if (elempack == 1 && out_elempack == 1)
    {
        // TODO deal with conv1x1s1d1
        if (is_conv1x1s1d1)
        {
            pipeline_convolution_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution_1x1s1d1->set_local_size_xyz(8, 1, std::min(8, num_output));
            pipeline_convolution_1x1s1d1->create(LayerShaderType::convolution_1x1s1d1, opt, specializations);
        }
        else
        {
            // TLOG_INFO("create pipeline pack1to1\n");
            pipeline_convolution = new Pipeline(vkdev);
            pipeline_convolution->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolution->create(LayerShaderType::convolution, opt, specializations);
        }
    }

    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        if (is_conv1x1s1d1)
        {
            pipeline_convolution_pack4_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution_pack4_1x1s1d1->set_local_size_xyz(8, 1, std::min(8, num_output / 4));
            pipeline_convolution_pack4_1x1s1d1->create(LayerShaderType::convolution_pack4_1x1s1d1, opt, specializations);
        }
        else if (is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
        {
            // winograd23
        }
        else
        {
            pipeline_convolution_pack4 = new Pipeline(vkdev);
            pipeline_convolution_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolution_pack4->create(LayerShaderType::convolution_pack4, opt, specializations);
        }
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        pipeline_convolution_pack1to4 = new Pipeline(vkdev);
        pipeline_convolution_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution_pack1to4->create(LayerShaderType::convolution_pack1to4, opt, specializations);
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        pipeline_convolution_pack4to1 = new Pipeline(vkdev);
        pipeline_convolution_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution_pack4to1->create(LayerShaderType::convolution_pack4to1, opt, specializations);
    }

    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        if (is_conv1x1s1d1)
        {
            pipeline_convolution_pack8_1x1s1d1 = new Pipeline(vkdev);
            pipeline_convolution_pack8_1x1s1d1->set_local_size_xyz(8, 1, std::min(8, num_output / 8));
            pipeline_convolution_pack8_1x1s1d1->create(LayerShaderType::convolution_pack8_1x1s1d1, opt, specializations);
        }
        else if (is_conv3x3s1d1 && num_input >= 16 && num_output >= 16)
        {
            // winograd23
        }
        else
        {
            pipeline_convolution_pack8 = new Pipeline(vkdev);
            pipeline_convolution_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_convolution_pack8->create(LayerShaderType::convolution_pack8, opt, specializations);
        }
    }

    // pack1to8
    if (elempack == 1 && out_elempack == 8)
    {
        pipeline_convolution_pack1to8 = new Pipeline(vkdev);
        pipeline_convolution_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution_pack1to8->create(LayerShaderType::convolution_pack1to8, opt, specializations);
    }

    // pack4to8
    if (elempack == 4 && out_elempack == 8)
    {
        pipeline_convolution_pack4to8 = new Pipeline(vkdev);
        pipeline_convolution_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution_pack4to8->create(LayerShaderType::convolution_pack4to8, opt, specializations);
    }

    // pack8to4
    if (elempack == 8 && out_elempack == 4)
    {
        pipeline_convolution_pack8to4 = new Pipeline(vkdev);
        pipeline_convolution_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution_pack8to4->create(LayerShaderType::convolution_pack8to4, opt, specializations);
    }

    // pack8to1
    if (elempack == 8 && out_elempack == 1)
    {
        pipeline_convolution_pack8to1 = new Pipeline(vkdev);
        pipeline_convolution_pack8to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolution_pack8to1->create(LayerShaderType::convolution_pack8to1, opt, specializations);
    }

    return 0;
}

int Convolution_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Convolution_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    tensor* weight_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);

    // Tensor weight_data = Tensor(weight_tensor->elem_num, 1, 1, weight_tensor->data);
    Tensor weight_data = Tensor(weight_tensor->elem_num, weight_tensor->data);

    // if (padding)
    // {
    //     padding->upload_model(cmd, opt);
    // }

    const int maxk = kernel_w * kernel_h;
    int num_output = output_c;
    int num_input = input_c; //weight_data_size / maxk / num_output;

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4
                                                                                       : 1;
    // int elempack = 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4
                                                                                             : 1;

    // TLOG_INFO("conv upload model pack:%d %d\n", elempack, out_elempack);

    Tensor weight_data_packed;
    {
        Tensor weight_data_r2 = weight_data.reshape(maxk, num_input, num_output);

        weight_data_packed.create(maxk, num_input / elempack, num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);
        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            Tensor g0 = weight_data_packed.channel(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                float* g00 = g0.row(p / elempack);

                for (int k = 0; k < maxk; k++)
                {
                    for (int i = 0; i < out_elempack; i++)
                    {
                        const Tensor k0 = weight_data_r2.channel(q + i);

                        for (int j = 0; j < elempack; j++)
                        {
                            const float* k00 = k0.row(p + j);

                            g00[0] = k00[k];

                            g00++;
                        }
                    }
                }
            }
        }
    }

    // ir_tensor* weight_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    // cmd.record_upload(weight_tensor, weight_data_gpu, opt);
    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);
    }

    // upload bias data
    if (node->input_num > 2)
    {
        tensor* bias_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);
        Tensor bias_data = Tensor(bias_tensor->elem_num, bias_tensor->data);

        // TLOG_INFO("bias data shape:%d %d %d\n", bias_data.c, bias_data.h, bias_data.w);

        Tensor bias_data_packed;
        convert_packing(bias_data, bias_data_packed, out_elempack);

        {
            cmd.record_upload(bias_data_packed, bias_data_gpu, opt);
        }
    }

    // if (innerproduct)
    // {
    //     innerproduct->upload_model(cmd, opt);
    // }

    return 0;
}

int Convolution_vulkan::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    // TLOG_INFO("in_c in_h in_w k_h k_w s p dilation group:%d %d %d %d %d %d %d %d %d\n", input_c, input_h, input_w, kernel_h, kernel_w, stride_h, pad_w0, dilation_h, group);
    VkTensor bottom_blob_dim3 = bottom_blob;
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        bottom_blob_dim3.dims = 3;
        bottom_blob_dim3.c = bottom_blob_dim3.w;
        bottom_blob_dim3.w = 1;
        bottom_blob_dim3.cstep = 1;
    }

    int w = bottom_blob_dim3.w;
    int h = bottom_blob_dim3.h;
    int channels = bottom_blob_dim3.c;
    size_t elemsize = bottom_blob_dim3.elemsize;
    int elempack = bottom_blob_dim3.elempack;
    // TLOG_INFO("botom shape:%d %d %d %d %d %d %d\n", bottom_blob.dims, bottom_blob.c, bottom_blob.h, bottom_blob.w, bottom_blob.elemsize, bottom_blob.elempack, bottom_blob.cstep);

    int out_elempack = opt.use_shader_pack8 && output_c % 8 == 0 ? 8 : output_c % 4 == 0 ? 4
                                                                                         : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    VkTensor bottom_blob_bordered = bottom_blob_dim3;
    if (pad_h0 > 0 || pad_h1 > 0 || pad_w0 > 0 || pad_w1 > 0)
    {
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->record_pipeline(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }

    // TLOG_INFO("forward convolution, w h c elemsize, elempack:%d %d %d %d %d\n", output_w, output_h, channels, elemsize, elempack);
    top_blob.create(output_w, output_h, output_c / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);

    // TLOG_INFO("convolution bottom shape:%d %d %d %d %d, top shape:%d %d %d %d %d\n", bottom_blob_bordered.dims, bottom_blob_bordered.w, bottom_blob_bordered.h, bottom_blob_bordered.c, bottom_blob_bordered.cstep, top_blob.dims, top_blob.w, top_blob.h, top_blob.c, top_blob.cstep);

    std::vector<VkTensor> bindings(4);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_bordered.dims;
    constants[1].i = bottom_blob_bordered.w;
    constants[2].i = bottom_blob_bordered.h;
    constants[3].i = bottom_blob_bordered.c;
    constants[4].i = bottom_blob_bordered.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    // record
    if (elempack == 1 && out_elempack == 1 && kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
    {
        VkTensor dispatcher;
        dispatcher.w = (top_blob.w * top_blob.h + 3) / 4;
        dispatcher.h = 1;
        dispatcher.c = top_blob.c;

        cmd.record_pipeline(pipeline_convolution_1x1s1d1, bindings, constants, dispatcher);
    }
    else if (elempack == 4 && out_elempack == 4 && kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
    {
        VkTensor dispatcher;
        dispatcher.w = (top_blob.w * top_blob.h + 3) / 4;
        dispatcher.h = 1;
        dispatcher.c = top_blob.c;

        cmd.record_pipeline(pipeline_convolution_pack4_1x1s1d1, bindings, constants, dispatcher);
    }
    else if (elempack == 8 && out_elempack == 8 && kernel_w == 1 && kernel_h == 1 && stride_w == 1 && stride_h == 1 && dilation_w == 1 && dilation_h == 1)
    {
        VkTensor dispatcher;
        dispatcher.w = (top_blob.w * top_blob.h + 3) / 4;
        dispatcher.h = 1;
        dispatcher.c = top_blob.c;

        cmd.record_pipeline(pipeline_convolution_pack8_1x1s1d1, bindings, constants, dispatcher);
    }
    else
    {
        const Pipeline* pipeline = 0;
        if (elempack == 1 && out_elempack == 1)
        {
            pipeline = pipeline_convolution;
        }
        else if (elempack == 4 && out_elempack == 4)
        {
            // TLOG_INFO("pipeline is pipeline_convolution_pack4\n");
            pipeline = pipeline_convolution_pack4;
        }
        else if (elempack == 1 && out_elempack == 4)
        {
            pipeline = pipeline_convolution_pack1to4;
        }
        else if (elempack == 4 && out_elempack == 1)
        {
            pipeline = pipeline_convolution_pack4to1;
        }
        else if (elempack == 8 && out_elempack == 8)
        {
            pipeline = pipeline_convolution_pack8;
        }
        else if (elempack == 1 && out_elempack == 8)
        {
            pipeline = pipeline_convolution_pack1to8;
        }
        else if (elempack == 4 && out_elempack == 8)
        {
            pipeline = pipeline_convolution_pack4to8;
        }
        else if (elempack == 8 && out_elempack == 4)
        {
            pipeline = pipeline_convolution_pack8to4;
        }
        else if (elempack == 8 && out_elempack == 1)
        {
            pipeline = pipeline_convolution_pack8to1;
        }

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    // TLOG_INFO("top shape:%d %d %d\n", top_blob.c, top_blob.h, top_blob.w);
    // cmd.record_pipeline(pipeline_convolution, bindings, constants, top_blob);
    // TLOG_INFO("run record convolution\n");
    return 0;
}

} // namespace TEngine
