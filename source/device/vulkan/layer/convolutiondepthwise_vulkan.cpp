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

#include "convolutiondepthwise_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

ConvolutionDepthWise_vulkan::ConvolutionDepthWise_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = true;
    padding = 0;
    pipeline_convolutiondepthwise = 0;
    pipeline_convolutiondepthwise_pack4 = 0;
    pipeline_convolutiondepthwise_pack8 = 0;
    graph = ir_graph;
    node = ir_node;

    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name = input->name;
    bottoms.push_back(name);

    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    name = output->name;
    tops.push_back(name);

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
    output_c = output->dims[1]; // param->output_channel;
    output_h = output->dims[2];
    output_w = output->dims[3];
}

int ConvolutionDepthWise_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    {
        padding = new Padding_vulkan(vkdev);

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

    // const int maxk = kernel_w * kernel_h;
    int channels = input_c; // (weight_data_size / group) / maxk / (num_output / group) * group;
    int num_output = output_c;

    int elempack = opt.use_shader_pack8 && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4
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

    std::vector<vk_specialization_type> specializations(11 + 10);
    specializations[0].i = kernel_w;                    // kernel_w;
    specializations[1].i = kernel_h;                    // kernel_h
    specializations[2].i = dilation_w;                  // dilation_w;
    specializations[3].i = dilation_h;                  // dilation_h;
    specializations[4].i = stride_w;                    // stride_w;
    specializations[5].i = stride_h;                    // stride_h;
    specializations[6].i = node->input_num > 2 ? 1 : 0; // bias_term;
    specializations[7].i = group;
    specializations[8].i = 1;      //param->activation;	// activation_type;
    specializations[9].f = 0;      //param->activation;	// activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[10].f = 0;     //param->activation; 	// activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[11 + 0].i = 0; // 3;	// shape_bordered_packed.dims;
    specializations[11 + 1].i = 0; // input_w + pad_w0 + pad_w1;	// shape_bordered_packed.w;
    specializations[11 + 2].i = 0; // input_h + pad_h0 + pad_h1;	// shape_bordered_packed.h;
    specializations[11 + 3].i = 0; // input_c;	// shape_bordered_packed.c;
    specializations[11 + 4].i = 0; // (input_w + pad_w0 + pad_w1) * (input_h + pad_h0 + pad_h1);	// shape_bordered_packed.cstep;
    specializations[11 + 5].i = 0; // 3;	// out_shape_packed.dims;
    specializations[11 + 6].i = 0; // output_w;	// out_shape_packed.w;
    specializations[11 + 7].i = 0; // output_h;	// out_shape_packed.h;
    specializations[11 + 8].i = 0; // output_c;	// out_shape_packed.c;
    specializations[11 + 9].i = 0; // output_w * output_h;	// out_shape_packed.cstep;

    VkTensor local_size_xyz;
    local_size_xyz.w = std::min(4, output_w);
    local_size_xyz.h = std::min(4, output_h);
    local_size_xyz.c = std::min(4, output_c);

    // pack1
    if (elempack == 1)
    {
        pipeline_convolutiondepthwise = new Pipeline(vkdev);
        pipeline_convolutiondepthwise->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise->create(LayerShaderType::convolutiondepthwise, opt, specializations);
    }

    // pack4
    if (elempack == 4)
    {
        pipeline_convolutiondepthwise_pack4 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_pack4->create(LayerShaderType::convolutiondepthwise_pack4, opt, specializations);
    }

    // pack8
    if (elempack == 8)
    {
        pipeline_convolutiondepthwise_pack8 = new Pipeline(vkdev);
        pipeline_convolutiondepthwise_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_convolutiondepthwise_pack8->create(LayerShaderType::convolutiondepthwise_pack8, opt, specializations);
    }

    return 0;
}

int ConvolutionDepthWise_vulkan::destroy_pipeline(const Option& opt)
{
    if (padding)
    {
        padding->destroy_pipeline(opt);
        delete padding;
        padding = 0;
    }

    delete pipeline_convolutiondepthwise;
    pipeline_convolutiondepthwise = 0;

    delete pipeline_convolutiondepthwise_pack4;
    pipeline_convolutiondepthwise_pack4 = 0;

    delete pipeline_convolutiondepthwise_pack8;
    pipeline_convolutiondepthwise_pack8 = 0;
    return 0;
}

int ConvolutionDepthWise_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    // upload kernel data
    const int maxk = kernel_w * kernel_h;
    int channels = input_c; // (weight_data_size / group) / maxk / (num_output / group) * group;
    int num_output = output_c;

    int elempack = opt.use_shader_pack8 && channels % 8 == 0 ? 8 : channels % 4 == 0 ? 4
                                                                                     : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4
                                                                                             : 1;

    tensor* weight_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    Tensor weight_data = Tensor(weight_tensor->elem_num, weight_tensor->data);

    Tensor weight_data_packed;
    Tensor weight_data_r2 = weight_data.reshape(maxk, group);
    TEngine::convert_packing(weight_data_r2, weight_data_packed, elempack);

    cmd.record_upload(weight_data_packed, weight_data_gpu, opt);

    // upload bias data
    if (node->input_num > 2)
    {
        tensor* bias_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);
        Tensor bias_data = Tensor(bias_tensor->elem_num, bias_tensor->data);
        Tensor bias_data_packed;
        convert_packing(bias_data, bias_data_packed, out_elempack);
        cmd.record_upload(bias_data_packed, bias_data_gpu, opt);
    }
    return 0;
}

int ConvolutionDepthWise_vulkan::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    VkTensor bottom_blob_bordered = bottom_blob;
    if (pad_h0 > 0 || pad_h1 > 0 || pad_w0 > 0 || pad_w1 > 0)
    {
        // bottom_blob_bordered.w = bottom_blob_bordered.w + pad_w0 + pad_w1;
        // bottom_blob_bordered.h = bottom_blob_bordered.h + pad_h0 + pad_h1;
        // bottom_blob_bordered.cstep = bottom_blob_bordered.w * bottom_blob_bordered.h;
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->record_pipeline(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }

    top_blob.create(output_w, output_h, output_c / elempack, elemsize, elempack, opt.blob_vkallocator);

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

    // printf("top shape:%d %d %d\n", top_blob.c, top_blob.h, top_blob.w);
    const Pipeline* pipeline = elempack == 8   ? pipeline_convolutiondepthwise_pack8
                               : elempack == 4 ? pipeline_convolutiondepthwise_pack4
                                               : pipeline_convolutiondepthwise;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace TEngine
