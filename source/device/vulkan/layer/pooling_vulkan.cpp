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

#include "pooling_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

Pooling_vulkan::Pooling_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = true;

    pipeline_pooling = 0;
    pipeline_pooling_pack4 = 0;
    pipeline_pooling_pack8 = 0;
    pipeline_pooling_global = 0;
    pipeline_pooling_global_pack4 = 0;
    pipeline_pooling_global_pack8 = 0;

    graph = ir_graph;
    node = ir_node;

    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name = input->name;
    bottoms.push_back(name);

    // Tensor* output_tensor = t_node->GetOutputTensor(0);
    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    name = output->name;
    tops.push_back(name);

    struct pool_param* param_ = (struct pool_param*)ir_node->op.param_mem;

    pooling_type = param_->pool_method; // 0:max    1:avg
    kernel_h = param_->kernel_h;
    kernel_w = param_->kernel_w;
    stride_h = param_->stride_h;
    stride_w = param_->stride_w;
    global = param_->global;
    caffe_flavor = param_->caffe_flavor;
    pad_h0 = param_->pad_h0;
    pad_w0 = param_->pad_w0;
    pad_h1 = param_->pad_h1;
    pad_w1 = param_->pad_w1;
    input_c = input->dims[1];
    input_h = input->dims[2];
    input_w = input->dims[3];
    output_c = output->dims[1];
    output_h = output->dims[2];
    output_w = output->dims[3];
    // printf("create pooling layer with param:%d %d %d %d %d %d %d %d %d %d\n", kernel_h, kernel_w, stride_h, stride_w, global, pad_h0, pad_h1, pad_w0, pad_w1, param_->alg);
}

int Pooling_vulkan::create_pipeline(const Option& opt)
{
    int elempack = opt.use_shader_pack8 && input_c % 8 == 0 ? 8 : input_c % 4 == 0 ? 4
                                                                                   : 1;
    int out_elempack = opt.use_shader_pack8 && output_c % 8 == 0 ? 8 : output_c % 4 == 0 ? 4
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

    if (global)
    {
        std::vector<vk_specialization_type> specializations(1 + 10);
        specializations[0].i = pooling_type;
        specializations[1 + 0].i = 3;
        specializations[1 + 1].i = input_w + pad_w0 + pad_w1;
        specializations[1 + 2].i = input_h + pad_h0 + pad_h1;
        specializations[1 + 3].i = input_c;
        specializations[1 + 4].i = (input_w + pad_w0 + pad_w1) * (input_h + pad_h0 + pad_h1);
        specializations[1 + 5].i = 3;
        specializations[1 + 6].i = output_c;
        specializations[1 + 7].i = output_h;
        specializations[1 + 8].i = output_w;
        specializations[1 + 9].i = output_h * output_w;

        VkTensor local_size_xyz;
        // if (out_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, output_w);
            local_size_xyz.h = std::min(4, output_h);
            local_size_xyz.c = std::min(4, output_c);
        }

        // pack1
        if (elempack == 1)
        {
            pipeline_pooling_global = new Pipeline(vkdev);
            pipeline_pooling_global->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_global->create(LayerShaderType::pooling_global, opt, specializations);
        }

        // pack4
        if (elempack == 4)
        {
            pipeline_pooling_global_pack4 = new Pipeline(vkdev);
            pipeline_pooling_global_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_global_pack4->create(LayerShaderType::pooling_global_pack4, opt, specializations);
        }

        // pack8
        if (opt.use_shader_pack8 || elempack == 8)
        {
            pipeline_pooling_global_pack8 = new Pipeline(vkdev);
            pipeline_pooling_global_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_global_pack8->create(LayerShaderType::pooling_global_pack8, opt, specializations);
        }
    }
    else
    {
        std::vector<vk_specialization_type> specializations(12 + 10);
        specializations[0].i = pooling_type;
        specializations[1].i = kernel_w;
        specializations[2].i = kernel_h;
        specializations[3].i = stride_w;
        specializations[4].i = stride_h;
        specializations[5].i = pad_w0;
        specializations[6].i = pad_w1;
        specializations[7].i = pad_h0;
        specializations[8].i = pad_h1;
        specializations[9].i = global;
        specializations[10].i = 0;     // pad_mode;
        specializations[11].i = 0;     // avgpool_count_include_pad;
        specializations[12 + 0].i = 0; // 3; // shape_bordered_packed.dims;
        specializations[12 + 1].i = 0; // input_w; // shape_bordered_packed.w;
        specializations[12 + 2].i = 0; // input_h; // shape_bordered_packed.h;
        specializations[12 + 3].i = 0; // input_c; // shape_bordered_packed.c;
        specializations[12 + 4].i = 0; // input_w * input_h; // shape_bordered_packed.cstep;
        specializations[12 + 5].i = 0; // 3; // out_shape_packed.dims;
        specializations[12 + 6].i = 0; // output_w; // out_shape_packed.w;
        specializations[12 + 7].i = 0; // output_h; // out_shape_packed.h;
        specializations[12 + 8].i = 0; // output_c; // out_shape_packed.c;
        specializations[12 + 9].i = 0; // output_h * output_c; // out_shape_packed.cstep;

        VkTensor local_size_xyz;
        local_size_xyz.w = std::min(4, output_w);
        local_size_xyz.h = std::min(4, output_h);
        local_size_xyz.c = std::min(4, output_c);

        // pack1
        if (elempack == 1)
        {
            pipeline_pooling = new Pipeline(vkdev);
            pipeline_pooling->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling->create(LayerShaderType::pooling, opt, specializations);
        }

        // pack4
        if (elempack == 4)
        {
            pipeline_pooling_pack4 = new Pipeline(vkdev);
            pipeline_pooling_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_pack4->create(LayerShaderType::pooling_pack4, opt, specializations);
        }

        // pack8
        if (opt.use_shader_pack8 || elempack == 8)
        {
            pipeline_pooling_pack8 = new Pipeline(vkdev);
            pipeline_pooling_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_pooling_pack8->create(LayerShaderType::pooling_pack8, opt, specializations);
        }
    }

    return 0;
}

int Pooling_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Pooling_vulkan::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    if (global)
    {
        // printf("input shape: %d %d %d, out shape: %d %d %d\n", input_c, input_h, input_w, output_c, output_h, output_w);
        top_blob.create(output_c / elempack, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
        // printf("top shape:%d %d %d\n", top_blob.c, top_blob.h, top_blob.w);
        std::vector<VkTensor> bindings(2);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_blob.dims;
        constants[1].i = bottom_blob.w;
        constants[2].i = bottom_blob.h;
        constants[3].i = bottom_blob.c;
        constants[4].i = bottom_blob.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;

        const Pipeline* pipeline = elempack == 8   ? pipeline_pooling_global_pack8
                                   : elempack == 4 ? pipeline_pooling_global_pack4
                                                   : pipeline_pooling_global;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);

        return 0;
    }

    VkTensor bottom_blob_bordered = bottom_blob;
    if (pad_h0 > 0 || pad_h1 > 0 || pad_w0 > 0 || pad_w1 > 0)
    {
        bottom_blob_bordered.w = bottom_blob_bordered.w + pad_w0 + pad_w1;
        bottom_blob_bordered.h = bottom_blob_bordered.h + pad_h0 + pad_h1;
        bottom_blob_bordered.cstep = bottom_blob_bordered.w * bottom_blob_bordered.h;
        Option opt_pad = opt;
        opt_pad.blob_vkallocator = opt.workspace_vkallocator;

        padding->record_pipeline(bottom_blob, bottom_blob_bordered, cmd, opt_pad);
    }

    top_blob.create(output_w, output_h, output_c / elempack, elemsize, elempack, opt.blob_vkallocator);

    std::vector<VkTensor> bindings(2);
    bindings[0] = bottom_blob_bordered;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
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
    constants[10].i = 0;
    constants[11].i = 0;

    const Pipeline* pipeline = elempack == 8   ? pipeline_pooling_pack8
                               : elempack == 4 ? pipeline_pooling_pack4
                                               : pipeline_pooling;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    return 0;
}

} // namespace TEngine
