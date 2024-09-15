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

#include "reshape_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

Reshape_vulkan::Reshape_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    permute_hwc = 0;
    permute_hc = 0;
    permute_hw = 0;
    permute_chw = 0;

    pipeline_reshape = 0;
    pipeline_reshape_pack4 = 0;
    pipeline_reshape_pack1to4 = 0;
    pipeline_reshape_pack4to1 = 0;
    pipeline_reshape_pack8 = 0;
    pipeline_reshape_pack1to8 = 0;
    pipeline_reshape_pack4to8 = 0;
    pipeline_reshape_pack8to4 = 0;
    pipeline_reshape_pack8to1 = 0;

    graph = ir_graph;
    node = ir_node;

    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name = input->name;
    bottoms.push_back(name);

    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    name = output->name;
    tops.push_back(name);

    // params
    input_c = input->dims[1]; // param->input_channel;
    input_h = input->dims[2];
    input_w = input->dims[3];

    struct reshape_param* param = (struct reshape_param*)ir_node->op.param_mem;

    ndim = param->dim_size;
    permute = param->reverse;
    // TODO fix
    // c = param->re_shape[0];
    // w = param->re_shape[1];
    // h = param->re_shape[2];
    if (param->dim_size == 4)
    {
        ndim = 3;
        output_c = output->dims[1]; // param->output_channel;
        output_h = output->dims[2];
        output_w = output->dims[3];

        c = output->dims[1]; // param->output_channel;
        h = output->dims[2];
        w = output->dims[3];
    }
    else
    {
        ndim = param->dim_size;

        output_c = output->dims[0]; // param->output_channel;
        output_h = output->dims[1];
        output_w = output->dims[2];

        c = output_c; // param->output_channel;
        h = output_h;
        w = output_w;
    }
}

int Reshape_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Tensor& shape = Tensor(input_w, input_h, input_c, (void*)0);        // bottom_shapes.empty() ? Tensor() : bottom_shapes[0];
    const Tensor& out_shape = Tensor(output_w, output_h, output_c, (void*)0); // top_shapes.empty() ? Tensor() : top_shapes[0];

    bool need_permute = permute == 1;
    if (shape.dims == 2 && ndim == 2 && shape.h == out_shape.h)
        need_permute = false;
    if (shape.dims == 3 && ndim == 3 && shape.c == out_shape.c)
        need_permute = false;

    Tensor shape_permuted = shape;
    Tensor out_shape_permuted = out_shape;
    if (need_permute)
    {
        if (shape.dims == 1) shape_permuted = Tensor(shape.w, (void*)0);
        if (shape.dims == 2) shape_permuted = Tensor(shape.h, shape.w, (void*)0);
        if (shape.dims == 3) shape_permuted = Tensor(shape.c, shape.w, shape.h, (void*)0);

        if (out_shape.dims == 1) out_shape_permuted = Tensor(out_shape.w, (void*)0);
        if (out_shape.dims == 2) out_shape_permuted = Tensor(out_shape.h, out_shape.w, (void*)0);
        if (out_shape.dims == 3) out_shape_permuted = Tensor(out_shape.c, out_shape.w, out_shape.h, (void*)0);
    }

    int elempack = 1;
    if (shape_permuted.dims == 1) elempack = opt.use_shader_pack8 && shape_permuted.w % 8 == 0 ? 8 : shape_permuted.w % 4 == 0 ? 4
                                                                                                                               : 1;
    if (shape_permuted.dims == 2) elempack = opt.use_shader_pack8 && shape_permuted.h % 8 == 0 ? 8 : shape_permuted.h % 4 == 0 ? 4
                                                                                                                               : 1;
    if (shape_permuted.dims == 3) elempack = opt.use_shader_pack8 && shape_permuted.c % 8 == 0 ? 8 : shape_permuted.c % 4 == 0 ? 4
                                                                                                                               : 1;

    int out_elempack = 1;
    if (out_shape_permuted.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape_permuted.w % 8 == 0 ? 8 : out_shape_permuted.w % 4 == 0 ? 4
                                                                                                                                               : 1;
    if (out_shape_permuted.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape_permuted.h % 8 == 0 ? 8 : out_shape_permuted.h % 4 == 0 ? 4
                                                                                                                                               : 1;
    if (out_shape_permuted.dims == 3) out_elempack = opt.use_shader_pack8 && out_shape_permuted.c % 8 == 0 ? 8 : out_shape_permuted.c % 4 == 0 ? 4
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

    Tensor shape_packed;
    if (shape_permuted.dims == 1) shape_packed = Tensor(shape_permuted.w / elempack, (void*)0, elemsize, elempack);
    if (shape_permuted.dims == 2) shape_packed = Tensor(shape_permuted.w, shape_permuted.h / elempack, (void*)0, elemsize, elempack);
    if (shape_permuted.dims == 3) shape_packed = Tensor(shape_permuted.w, shape_permuted.h, shape_permuted.c / elempack, (void*)0, elemsize, elempack);

    Tensor out_shape_packed;
    if (out_shape_permuted.dims == 1) out_shape_packed = Tensor(out_shape_permuted.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape_permuted.dims == 2) out_shape_packed = Tensor(out_shape_permuted.w, out_shape_permuted.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape_permuted.dims == 3) out_shape_packed = Tensor(out_shape_permuted.w, out_shape_permuted.h, out_shape_permuted.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    // check blob shape
    {
        opt.use_image_storage = false;
    }

    std::vector<vk_specialization_type> specializations(1 + 10);
    specializations[0].i = ndim;
    specializations[1 + 0].i = 0; // shape_packed.dims;
    specializations[1 + 1].i = 0; // shape_packed.w;
    specializations[1 + 2].i = 0; // shape_packed.h;
    specializations[1 + 3].i = 0; // shape_packed.c;
    specializations[1 + 4].i = 0; // shape_packed.cstep;
    specializations[1 + 5].i = 0; // out_shape_packed.dims;
    specializations[1 + 6].i = 0; // out_shape_packed.w;
    specializations[1 + 7].i = 0; // out_shape_packed.h;
    specializations[1 + 8].i = 0; // out_shape_packed.c;
    specializations[1 + 9].i = 0; // out_shape_packed.cstep;

    Tensor local_size_xyz_bottom; // pack4to1 and pack8to1
    if (shape_packed.dims == 1)
    {
        local_size_xyz_bottom.w = std::min(64, shape_packed.w);
        local_size_xyz_bottom.h = 1;
        local_size_xyz_bottom.c = 1;
    }
    if (shape_packed.dims == 2)
    {
        local_size_xyz_bottom.w = std::min(8, shape_packed.w);
        local_size_xyz_bottom.h = std::min(8, shape_packed.h);
        local_size_xyz_bottom.c = 1;
    }
    if (shape_packed.dims == 3)
    {
        local_size_xyz_bottom.w = std::min(4, shape_packed.w);
        local_size_xyz_bottom.h = std::min(4, shape_packed.h);
        local_size_xyz_bottom.c = std::min(4, shape_packed.c);
    }

    Tensor local_size_xyz;
    if (out_shape_packed.dims == 1)
    {
        local_size_xyz.w = std::min(64, out_shape_packed.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (out_shape_packed.dims == 2)
    {
        local_size_xyz.w = std::min(8, out_shape_packed.w);
        local_size_xyz.h = std::min(8, out_shape_packed.h);
        local_size_xyz.c = 1;
    }
    if (out_shape_packed.dims == 3)
    {
        local_size_xyz.w = std::min(4, out_shape_packed.w);
        local_size_xyz.h = std::min(4, out_shape_packed.h);
        local_size_xyz.c = std::min(4, out_shape_packed.c);
    }

    // pack1
    if (shape_permuted.dims == 0 || (elempack == 1 && out_elempack == 1))
    {
        pipeline_reshape = new Pipeline(vkdev);
        pipeline_reshape->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape->create(LayerShaderType::reshape, opt, specializations);
    }

    // pack4
    if (shape_permuted.dims == 0 || (elempack == 4 && out_elempack == 4))
    {
        pipeline_reshape_pack4 = new Pipeline(vkdev);
        pipeline_reshape_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack4->create(LayerShaderType::reshape_pack4, opt, specializations);
    }

    // pack1to4
    if (shape_permuted.dims == 0 || (elempack == 1 && out_elempack == 4))
    {
        pipeline_reshape_pack1to4 = new Pipeline(vkdev);
        pipeline_reshape_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack1to4->create(LayerShaderType::reshape_pack1to4, opt, specializations);
    }

    // pack4to1
    if (shape_permuted.dims == 0 || (elempack == 4 && out_elempack == 1))
    {
        pipeline_reshape_pack4to1 = new Pipeline(vkdev);
        pipeline_reshape_pack4to1->set_optimal_local_size_xyz(local_size_xyz_bottom);
        pipeline_reshape_pack4to1->create(LayerShaderType::reshape_pack4to1, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 8 && out_elempack == 8))
    {
        pipeline_reshape_pack8 = new Pipeline(vkdev);
        pipeline_reshape_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack8->create(LayerShaderType::reshape_pack8, opt, specializations);
    }

    // pack1to8
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 1 && out_elempack == 8))
    {
        pipeline_reshape_pack1to8 = new Pipeline(vkdev);
        pipeline_reshape_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack1to8->create(LayerShaderType::reshape_pack1to8, opt, specializations);
    }

    // pack4to8
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 4 && out_elempack == 8))
    {
        pipeline_reshape_pack4to8 = new Pipeline(vkdev);
        pipeline_reshape_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack4to8->create(LayerShaderType::reshape_pack4to8, opt, specializations);
    }

    // pack8to4
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 8 && out_elempack == 4))
    {
        pipeline_reshape_pack8to4 = new Pipeline(vkdev);
        pipeline_reshape_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_reshape_pack8to4->create(LayerShaderType::reshape_pack8to4, opt, specializations);
    }

    // pack8to1
    if ((opt.use_shader_pack8 && shape_permuted.dims == 0) || (elempack == 8 && out_elempack == 1))
    {
        pipeline_reshape_pack8to1 = new Pipeline(vkdev);
        pipeline_reshape_pack8to1->set_optimal_local_size_xyz(local_size_xyz_bottom);
        pipeline_reshape_pack8to1->create(LayerShaderType::reshape_pack8to1, opt, specializations);
    }

    return 0;
}

int Reshape_vulkan::destroy_pipeline(const Option& opt)
{
    if (permute_hwc)
    {
        permute_hwc->destroy_pipeline(opt);
        delete permute_hwc;
        permute_hwc = 0;
    }

    if (permute_hc)
    {
        permute_hc->destroy_pipeline(opt);
        delete permute_hc;
        permute_hc = 0;
    }

    if (permute_hw)
    {
        permute_hw->destroy_pipeline(opt);
        delete permute_hw;
        permute_hw = 0;
    }

    if (permute_chw)
    {
        permute_chw->destroy_pipeline(opt);
        delete permute_chw;
        permute_chw = 0;
    }

    delete pipeline_reshape;
    pipeline_reshape = 0;

    delete pipeline_reshape_pack4;
    pipeline_reshape_pack4 = 0;

    delete pipeline_reshape_pack1to4;
    pipeline_reshape_pack1to4 = 0;

    delete pipeline_reshape_pack4to1;
    pipeline_reshape_pack4to1 = 0;

    delete pipeline_reshape_pack8;
    pipeline_reshape_pack8 = 0;

    delete pipeline_reshape_pack1to8;
    pipeline_reshape_pack1to8 = 0;

    delete pipeline_reshape_pack4to8;
    pipeline_reshape_pack4to8 = 0;

    delete pipeline_reshape_pack8to4;
    pipeline_reshape_pack8to4 = 0;

    delete pipeline_reshape_pack8to1;
    pipeline_reshape_pack8to1 = 0;

    return 0;
}

int Reshape_vulkan::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;
    int out_elempack;

    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c * elempack;

    // resolve out shape
    int outw = w;
    int outh = h;
    int outc = c;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;

        if (outw == -1)
            outw = total;

        out_elempack = opt.use_shader_pack8 && outw % 8 == 0 ? 8 : outw % 4 == 0 ? 4
                                                                                 : 1;

        if (dims == 1 && bottom_blob.w == outw && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 2)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        out_elempack = opt.use_shader_pack8 && outh % 8 == 0 ? 8 : outh % 4 == 0 ? 4
                                                                                 : 1;

        if (dims == 2 && bottom_blob.h == outh && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }

    if (ndim == 3)
    {
        if (outw == 0)
            outw = dims == 1 ? bottom_blob.w * elempack : bottom_blob.w;
        if (outh == 0)
            outh = dims == 2 ? bottom_blob.h * elempack : bottom_blob.h;
        if (outc == 0)
            outc = dims == 3 ? bottom_blob.c * elempack : bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;

        out_elempack = opt.use_shader_pack8 && outc % 8 == 0 ? 8 : outc % 4 == 0 ? 4
                                                                                 : 1;

        if (dims == 3 && bottom_blob.c == outc && elempack == out_elempack)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            return 0;
        }
    }

    bool need_permute = permute == 1;
    if (dims == 2 && ndim == 2 && bottom_blob.h * elempack == outh)
        need_permute = false;
    if (dims == 3 && ndim == 3 && bottom_blob.c * elempack == outc)
        need_permute = false;

    if (ndim == 1)
    {
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        top_blob.create(outw / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (ndim == 2)
    {
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        top_blob.create(outw, outh / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }
    if (ndim == 3)
    {
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        top_blob.create(outw, outh, outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    }

    if (top_blob.empty())
        return -100;

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

    if (elempack == 1 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack4, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack1to4, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape_pack4to1, bindings, constants, bottom_blob);
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_reshape_pack8, bindings, constants, top_blob);
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_reshape_pack1to8, bindings, constants, top_blob);
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        cmd.record_pipeline(pipeline_reshape_pack4to8, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        cmd.record_pipeline(pipeline_reshape_pack8to4, bindings, constants, top_blob);
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        cmd.record_pipeline(pipeline_reshape_pack8to1, bindings, constants, bottom_blob);
    }

    return 0;
}

} // namespace TEngine
