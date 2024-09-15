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

#include "padding_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

Padding_vulkan::Padding_vulkan(const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = true;
    pipeline_padding = 0;
    pipeline_padding_pack4 = 0;
    pipeline_padding_pack8 = 0;
}

int Padding_vulkan::create_pipeline(const Option& opt)
{
    int elempack = 1;
    elempack = opt.use_shader_pack8 && input_c % 8 == 0 ? 8 : input_c % 4 == 0 ? 4
                                                                               : 1;
    int out_elempack;
    out_elempack = opt.use_shader_pack8 && output_c % 8 == 0 ? 8 : output_c % 4 == 0 ? 4
                                                                                     : 1;

    // printf("create padding pipeline elempack:%d %d \n", elempack, out_elempack);

    std::vector<vk_specialization_type> specializations(3 + 10);
    specializations[0].i = type;
    specializations[1].f = value;
    specializations[2].i = 0;                       // per_channel_pad_data_size ? 1 : 0;
    specializations[3 + 0].i = 3;                   // shape_packed.dims;
    specializations[3 + 1].i = input_w;             // shape_packed.w;
    specializations[3 + 2].i = input_h;             // shape_packed.h;
    specializations[3 + 3].i = input_c;             // shape_packed.c;
    specializations[3 + 4].i = input_w * input_h;   // shape_packed.cstep;
    specializations[3 + 5].i = 3;                   // out_shape_packed.dims;
    specializations[3 + 6].i = output_w;            // out_shape_packed.w;
    specializations[3 + 7].i = output_h;            // out_shape_packed.h;
    specializations[3 + 8].i = output_c;            // out_shape_packed.c;
    specializations[3 + 9].i = output_w * output_h; // out_shape_packed.cstep;

    VkTensor local_size_xyz;
    // if (out_shape_packed.dims != 0)
    {
        local_size_xyz.w = std::min(4, output_w);
        local_size_xyz.h = std::min(4, output_h);
        local_size_xyz.c = std::min(4, output_c);
    }

    // pack1
    // if (shape.dims == 0 || elempack == 1)
    if (elempack == 1)
    {
        pipeline_padding = new Pipeline(vkdev);
        pipeline_padding->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_padding->create(LayerShaderType::padding, opt, specializations);
    }

    // pack4
    // if (shape.dims == 0 || elempack == 4)
    if (elempack == 4)
    {
        pipeline_padding_pack4 = new Pipeline(vkdev);
        pipeline_padding_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_padding_pack4->create(LayerShaderType::padding_pack4, opt, specializations);
    }

    // pack8
    // if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
    if (opt.use_shader_pack8 || elempack == 8)
    {
        pipeline_padding_pack8 = new Pipeline(vkdev);
        pipeline_padding_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_padding_pack8->create(LayerShaderType::padding_pack8, opt, specializations);
    }

    return 0;
}

int Padding_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Padding_vulkan::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = w + left + right;
    int outh = h + top + bottom;

    // printf("create padding top_blob vktensor, w, h, c:%d %d %d\n", outw, outh, channels);
    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkTensor> bindings(2);
    bindings[0] = bottom_blob;
    bindings[1] = top_blob;

    std::vector<vk_constant_type> constants(12);
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
    constants[10].i = left;
    constants[11].i = top;

    // printf("padding shape:%d %d %d %d %d %d %d %d %d\n", top_blob.c, top_blob.h, top_blob.w, top_blob.cstep, bottom_blob.c, bottom_blob.h, bottom_blob.w, bottom_blob.cstep, elempack);
    const Pipeline* pipeline = elempack == 8   ? pipeline_padding_pack8
                               : elempack == 4 ? pipeline_padding_pack4
                                               : pipeline_padding;

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace TEngine
