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

#include "packing_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

Packing_vulkan::Packing_vulkan(const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = true;
    pipeline_packing = 0;
    pipeline_packing_pack4 = 0;
    pipeline_packing_pack8 = 0;
    pipeline_packing_pack1to4 = 0;
    pipeline_packing_pack4to1 = 0;
    pipeline_packing_pack1to8 = 0;
    pipeline_packing_pack4to8 = 0;
    pipeline_packing_pack8to4 = 0;
    pipeline_packing_pack8to1 = 0;
}

int Packing_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    // const Mat& shape = bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    // const Mat& out_shape = top_shapes.empty() ? Mat() : top_shapes[0];

    size_t out_elemsize;
    if (opt.use_fp16_storage)
    {
        out_elemsize = out_elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        out_elemsize = out_elempack == 1 ? 4u : out_elempack * 2u;
    }
    else
    {
        out_elemsize = out_elempack * 4u;
    }

    // type casting override
    if (cast_type_to == 1)
    {
        out_elemsize = out_elempack * 4u;
    }

    // Mat out_shape_packed;
    // if (out_shape.dims == 1) out_shape_packed = Mat(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    // if (out_shape.dims == 2) out_shape_packed = Mat(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    // if (out_shape.dims == 3) out_shape_packed = Mat(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    {
        opt.use_image_storage = false;
    }

    std::vector<vk_specialization_type> specializations(2 + 10);
    specializations[0].i = storage_type_from;
    specializations[1].i = storage_type_to;
    specializations[2 + 0].i = 0; // FIXME shape elempack may be dynamic
    specializations[2 + 1].i = 0;
    specializations[2 + 2].i = 0;
    specializations[2 + 3].i = 0;
    specializations[2 + 4].i = 0;
    specializations[2 + 5].i = 0; //out_shape_packed_dims;
    specializations[2 + 6].i = 0; //out_shape_packed_w;
    specializations[2 + 7].i = 0; //out_shape_packed_h;
    specializations[2 + 8].i = 0; //out_shape_packed_c;
    specializations[2 + 9].i = 0; //out_shape_packed_cstep;

    // printf("out shape dims:%d ---------------------------------\n", out_shape_packed_dims);

    VkTensor local_size_xyz; // TODO more precise group size guessed from out_shape_packed
    if (out_shape_packed_dims == 1)
    {
        local_size_xyz.w = 64;
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (out_shape_packed_dims == 2)
    {
        local_size_xyz.w = 8;
        local_size_xyz.h = 8;
        local_size_xyz.c = 1;
    }
    if (out_shape_packed_dims == 3)
    {
        local_size_xyz.w = 4;
        local_size_xyz.h = 4;
        local_size_xyz.c = 4;
    }

    if (out_elempack == 8)
    {
        pipeline_packing_pack8 = new Pipeline(vkdev);
        pipeline_packing_pack8->set_optimal_local_size_xyz(local_size_xyz);

        pipeline_packing_pack1to8 = new Pipeline(vkdev);
        pipeline_packing_pack1to8->set_optimal_local_size_xyz(local_size_xyz);

        pipeline_packing_pack4to8 = new Pipeline(vkdev);
        pipeline_packing_pack4to8->set_optimal_local_size_xyz(local_size_xyz);

        if (cast_type_from == cast_type_to)
        {
            pipeline_packing_pack8->create(LayerShaderType::packing_pack8, opt, specializations);
            pipeline_packing_pack1to8->create(LayerShaderType::packing_pack1to8, opt, specializations);
            pipeline_packing_pack4to8->create(LayerShaderType::packing_pack4to8, opt, specializations);
        }
        else if (cast_type_from == 1)
        {
            pipeline_packing_pack8->create(LayerShaderType::packing_pack8_fp32_to_fp16, opt, specializations);
            pipeline_packing_pack1to8->create(LayerShaderType::packing_pack1to8_fp32_to_fp16, opt, specializations);
            pipeline_packing_pack4to8->create(LayerShaderType::packing_pack4to8_fp32_to_fp16, opt, specializations);
        }
        else if (cast_type_to == 1)
        {
            pipeline_packing_pack8->create(LayerShaderType::packing_pack8_fp16_to_fp32, opt, specializations);
            pipeline_packing_pack1to8->create(LayerShaderType::packing_pack1to8_fp16_to_fp32, opt, specializations);
            pipeline_packing_pack4to8->create(LayerShaderType::packing_pack4to8_fp16_to_fp32, opt, specializations);
        }
    }

    if (out_elempack == 4)
    {
        pipeline_packing_pack4 = new Pipeline(vkdev);
        pipeline_packing_pack4->set_optimal_local_size_xyz(local_size_xyz);

        pipeline_packing_pack1to4 = new Pipeline(vkdev);
        pipeline_packing_pack1to4->set_optimal_local_size_xyz(local_size_xyz);

        pipeline_packing_pack8to4 = new Pipeline(vkdev);
        pipeline_packing_pack8to4->set_optimal_local_size_xyz(local_size_xyz);

        if (cast_type_from == cast_type_to)
        {
            pipeline_packing_pack4->create(LayerShaderType::packing_pack4, opt, specializations);
            pipeline_packing_pack1to4->create(LayerShaderType::packing_pack1to4, opt, specializations);
            pipeline_packing_pack8to4->create(LayerShaderType::packing_pack8to4, opt, specializations);
        }
        else if (cast_type_from == 1)
        {
            pipeline_packing_pack4->create(LayerShaderType::packing_pack4_fp32_to_fp16, opt, specializations);
            pipeline_packing_pack1to4->create(LayerShaderType::packing_pack1to4_fp32_to_fp16, opt, specializations);
            pipeline_packing_pack8to4->create(LayerShaderType::packing_pack8to4_fp32_to_fp16, opt, specializations);
        }
        else if (cast_type_to == 1)
        {
            pipeline_packing_pack4->create(LayerShaderType::packing_pack4_fp16_to_fp32, opt, specializations);
            pipeline_packing_pack1to4->create(LayerShaderType::packing_pack1to4_fp16_to_fp32, opt, specializations);
            pipeline_packing_pack8to4->create(LayerShaderType::packing_pack8to4_fp16_to_fp32, opt, specializations);
        }
    }

    if (out_elempack == 1)
    {
        pipeline_packing = new Pipeline(vkdev);
        pipeline_packing->set_optimal_local_size_xyz(local_size_xyz);

        pipeline_packing_pack4to1 = new Pipeline(vkdev);
        pipeline_packing_pack4to1->set_optimal_local_size_xyz(local_size_xyz);

        pipeline_packing_pack8to1 = new Pipeline(vkdev);
        pipeline_packing_pack8to1->set_optimal_local_size_xyz(local_size_xyz);

        if (cast_type_from == cast_type_to)
        {
            pipeline_packing->create(LayerShaderType::packing, opt, specializations);
            pipeline_packing_pack4to1->create(LayerShaderType::packing_pack4to1, opt, specializations);
            pipeline_packing_pack8to1->create(LayerShaderType::packing_pack8to1, opt, specializations);
        }
        else if (cast_type_from == 1)
        {
            pipeline_packing->create(LayerShaderType::packing_fp32_to_fp16, opt, specializations);
            pipeline_packing_pack4to1->create(LayerShaderType::packing_pack4to1_fp32_to_fp16, opt, specializations);
            pipeline_packing_pack8to1->create(LayerShaderType::packing_pack8to1_fp32_to_fp16, opt, specializations);
        }
        else if (cast_type_to == 1)
        {
            pipeline_packing->create(LayerShaderType::packing_fp16_to_fp32, opt, specializations);
            pipeline_packing_pack4to1->create(LayerShaderType::packing_pack4to1_fp16_to_fp32, opt, specializations);
            pipeline_packing_pack8to1->create(LayerShaderType::packing_pack8to1_fp16_to_fp32, opt, specializations);
        }
    }

    return 0;
}

int Packing_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_packing;
    pipeline_packing = 0;

    delete pipeline_packing_pack4;
    pipeline_packing_pack4 = 0;

    delete pipeline_packing_pack8;
    pipeline_packing_pack8 = 0;

    delete pipeline_packing_pack1to4;
    pipeline_packing_pack1to4 = 0;

    delete pipeline_packing_pack4to1;
    pipeline_packing_pack4to1 = 0;

    delete pipeline_packing_pack1to8;
    pipeline_packing_pack1to8 = 0;

    delete pipeline_packing_pack4to8;
    pipeline_packing_pack4to8 = 0;

    delete pipeline_packing_pack8to4;
    pipeline_packing_pack8to4 = 0;

    delete pipeline_packing_pack8to1;
    pipeline_packing_pack8to1 = 0;

    return 0;
}

int Packing_vulkan::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    int elempack = bottom_blob.elempack;
    // printf("Packing_vulkan b2b %d %d   %d %d   %d %d\n", elempack, out_elempack, cast_type_from, cast_type_to, storage_type_from, storage_type_to);

    if (elempack == out_elempack && cast_type_from == cast_type_to && bottom_blob.allocator == opt.blob_vkallocator)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    if (!use_padding)
    {
        // identity if use_padding not allowed
        if (dims == 1 && w * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 2 && h * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
        if (dims == 3 && channels * elempack % out_elempack != 0)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    size_t out_elemsize;
    if (cast_type_to == 0)
    {
        if (opt.use_fp16_storage)
        {
            out_elemsize = out_elempack * 2u;
        }
        else if (opt.use_fp16_packed)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }
        else
        {
            out_elemsize = out_elempack * 4u;
        }
    }
    else if (cast_type_to == 1)
    {
        out_elemsize = out_elempack * 4u;
    }
    else if (cast_type_to == 2)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }
    else // if (cast_type_to == 3)
    {
        out_elemsize = out_elempack * 2u;
    }

    if (dims == 1)
    {
        if (opt.use_fp16_storage && out_elempack == 1 && cast_type_from == cast_type_to && bottom_blob.allocator == opt.blob_vkallocator)
        {
            top_blob = bottom_blob;
            top_blob.w = w * elempack;
            top_blob.cstep = w * elempack;
            top_blob.elemsize = elemsize / elempack;
            top_blob.elempack = out_elempack;
            return 0;
        }

        int outw = (w * elempack + out_elempack - 1) / out_elempack;

        top_blob.create(outw, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
        // int outw = (w * elempack + out_elempack - 1) / out_elempack;

        // if (opt.use_fp16_packed && !opt.use_fp16_storage)
        // {
        //     if (out_elempack == 8) out_elemsize = 8*2u;
        //     if (out_elempack == 4) out_elemsize = 4*2u;
        //     if (out_elempack == 1) out_elemsize = 4u;
        // }

        // // type casting override
        // if (cast_type_to == 1)
        // {
        //     out_elemsize = out_elempack * 4u;
        // }

        // top_blob.create(outw, out_elemsize, out_elempack, opt.blob_vkallocator);
        // if (top_blob.empty())
        //     return -100;
    }

    if (dims == 2)
    {
        int outh = (h * elempack + out_elempack - 1) / out_elempack;

        top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        // int outh = (h * elempack + out_elempack - 1) / out_elempack;
        // size_t out_elemsize = elemsize / elempack * out_elempack;
        // if (opt.use_fp16_packed && !opt.use_fp16_storage)
        // {
        //     if (out_elempack == 8) out_elemsize = 8*2u;
        //     if (out_elempack == 4) out_elemsize = 4*2u;
        //     if (out_elempack == 1) out_elemsize = 4u;
        // }

        // // type casting override
        // if (cast_type_to == 1)
        // {
        //     out_elemsize = out_elempack * 4u;
        // }

        // top_blob.create(w, outh, out_elemsize, out_elempack, opt.blob_vkallocator);
        // if (top_blob.empty())
        //     return -100;
    }

    if (dims == 3)
    {
        int outc = (channels * elempack + out_elempack - 1) / out_elempack;

        top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;
        // int outc = (channels * elempack + out_elempack - 1) / out_elempack;
        // size_t out_elemsize = elemsize / elempack * out_elempack;
        // if (opt.use_fp16_packed && !opt.use_fp16_storage)
        // {
        //     if (out_elempack == 8) out_elemsize = 8*2u;
        //     if (out_elempack == 4) out_elemsize = 4*2u;
        //     if (out_elempack == 1) out_elemsize = 4u;
        // }

        // // type casting override
        // if (cast_type_to == 1)
        // {
        //     out_elemsize = out_elempack * 4u;
        // }

        // top_blob.create(w, h, outc, out_elemsize, out_elempack, opt.blob_vkallocator);
        // if (top_blob.empty())
        //     return -100;
    }

    std::vector<VkTensor> buffer_bindings(2);
    buffer_bindings[0] = bottom_blob;
    buffer_bindings[1] = top_blob;

    std::vector<VkImageTensor> image_bindings(2);
    if (!opt.use_image_storage)
    {
        image_bindings.clear();
    }

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

    // printf("record packing pipeline:%d %d %d %d %d %d %d %d %d\n", top_blob.dims, top_blob.c, top_blob.h, top_blob.w, top_blob.cstep, top_blob.elempack, top_blob.elemsize, elempack, out_elempack);

    if (elempack == 1 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_packing, buffer_bindings, image_bindings, constants, top_blob);
    }
    if (elempack == 4 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_packing_pack4, buffer_bindings, image_bindings, constants, top_blob);
    }
    if (elempack == 1 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_packing_pack1to4, buffer_bindings, image_bindings, constants, top_blob);
    }
    if (elempack == 4 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_packing_pack4to1, buffer_bindings, image_bindings, constants, bottom_blob);
    }
    if (elempack == 8 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_packing_pack8, buffer_bindings, image_bindings, constants, top_blob);
    }
    if (elempack == 1 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_packing_pack1to8, buffer_bindings, image_bindings, constants, top_blob);
    }
    if (elempack == 4 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_packing_pack4to8, buffer_bindings, image_bindings, constants, top_blob);
    }
    if (elempack == 8 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_packing_pack8to4, buffer_bindings, image_bindings, constants, bottom_blob);
    }
    if (elempack == 8 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_packing_pack8to1, buffer_bindings, image_bindings, constants, bottom_blob);
    }

    // printf("run packing vulkan record pipeline\n");
    return 0;
}

} // namespace TEngine
