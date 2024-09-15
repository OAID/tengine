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

#include "crop_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

Crop_vulkan::Crop_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = true;
    pipeline_crop = 0;
    pipeline_crop_pack4 = 0;
    pipeline_crop_pack1to4 = 0;
    pipeline_crop_pack4to1 = 0;
    pipeline_crop_pack8 = 0;
    pipeline_crop_pack1to8 = 0;
    pipeline_crop_pack4to8 = 0;
    pipeline_crop_pack8to4 = 0;
    pipeline_crop_pack8to1 = 0;

    graph = ir_graph;
    node = ir_node;

    for (int i = 0; i < ir_node->input_num; i++)
    {
        struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[i]);
        std::string name = input->name;
        bottoms.push_back(name);
    }

    for (int i = 0; i < ir_node->output_num; i++)
    {
        struct tensor* output = get_ir_graph_tensor(graph, node->input_tensors[i]);
        std::string name = output->name;
        tops.push_back(name);
    }

    // params
    struct tensor* input_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    input_c = input_tensor->dims[1]; // param->input_channel;
    input_h = input_tensor->dims[2];
    input_w = input_tensor->dims[3];
    output_c = output_tensor->dims[1]; // param->output_channel;
    output_h = output_tensor->dims[2];
    output_w = output_tensor->dims[3];

    struct crop_param* param = (struct crop_param*)ir_node->op.param_mem;

    int num_args = param->num_args;
    int offset_c = 0; // param->offset_c;
    int offset_h = 0; // param->offset_h;
    int offset_w = 0; // param->offset_w;
    int crop_h = param->crop_h;
    int crop_w = param->crop_w;
    int center_crop = param->center_crop;
    int axis = param->axis;
    int flag = param->flag;
}

int Crop_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;

    const Tensor& shape = Tensor(input_w, input_h, input_c, (void*)0);        // bottom_shapes.empty() ? Tensor() : bottom_shapes[0];
    const Tensor& out_shape = Tensor(output_w, output_h, output_c, (void*)0); // top_shapes.empty() ? Tensor() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4
                                                                                                    : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4
                                                                                                    : 1;
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4
                                                                                                    : 1;

    int out_elempack = 1;
    if (out_shape.dims == 1) out_elempack = opt.use_shader_pack8 && out_shape.w % 8 == 0 ? 8 : out_shape.w % 4 == 0 ? 4
                                                                                                                    : 1;
    if (out_shape.dims == 2) out_elempack = opt.use_shader_pack8 && out_shape.h % 8 == 0 ? 8 : out_shape.h % 4 == 0 ? 4
                                                                                                                    : 1;
    if (out_shape.dims == 3) out_elempack = opt.use_shader_pack8 && out_shape.c % 8 == 0 ? 8 : out_shape.c % 4 == 0 ? 4
                                                                                                                    : 1;

    int offset_elempack = 1;

    {
        // TODO vec and image crop
        if (offset_c == 0)
            offset_elempack = elempack;
        else
            offset_elempack = opt.use_shader_pack8 && offset_c % 8 == 0 ? 8 : offset_c % 4 == 0 ? 4
                                                                                                : 1;
    }

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
    if (shape.dims == 1) shape_packed = Tensor(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Tensor(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Tensor(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    Tensor out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Tensor(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 2) out_shape_packed = Tensor(out_shape.w, out_shape.h / out_elempack, (void*)0, out_elemsize, out_elempack);
    if (out_shape.dims == 3) out_shape_packed = Tensor(out_shape.w, out_shape.h, out_shape.c / out_elempack, (void*)0, out_elemsize, out_elempack);

    Tensor shape_unpacked = shape_packed;
    if (bottoms.size() == 1 && shape.dims != 0 && elempack == out_elempack && elempack > offset_elempack)
    {
        size_t offset_elemsize;
        if (opt.use_fp16_storage)
        {
            offset_elemsize = offset_elempack * 2u;
        }
        else if (opt.use_fp16_packed)
        {
            offset_elemsize = offset_elempack == 1 ? 4u : offset_elempack * 2u;
        }
        else
        {
            offset_elemsize = offset_elempack * 4u;
        }

        if (shape.dims == 1) shape_unpacked = Tensor(shape.w / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
        if (shape.dims == 2) shape_unpacked = Tensor(shape.w, shape.h / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
        if (shape.dims == 3) shape_unpacked = Tensor(shape.w, shape.h, shape.c / offset_elempack, (void*)0, offset_elemsize, offset_elempack);
    }

    std::vector<vk_specialization_type> specializations(1 + 10);
    specializations[0].i = vkdev->info.bug_implicit_fp16_arithmetic;
    specializations[1 + 0].i = 0; // shape_unpacked.dims;
    specializations[1 + 1].i = 0; // shape_unpacked.w;
    specializations[1 + 2].i = 0; // shape_unpacked.h;
    specializations[1 + 3].i = 0; // shape_unpacked.c;
    specializations[1 + 4].i = 0; // shape_unpacked.cstep;
    specializations[1 + 5].i = 0; // out_shape_packed.dims;
    specializations[1 + 6].i = 0; // out_shape_packed.w;
    specializations[1 + 7].i = 0; // out_shape_packed.h;
    specializations[1 + 8].i = 0; // out_shape_packed.c;
    specializations[1 + 9].i = 0; // out_shape_packed.cstep;

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
    if (out_shape.dims == 0 || out_elempack == 1)
    {
        pipeline_crop = new Pipeline(vkdev);
        pipeline_crop->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop->create(LayerShaderType::crop, opt, specializations);
    }

    // pack4
    if (out_shape.dims == 0 || out_elempack == 4)
    {
        pipeline_crop_pack4 = new Pipeline(vkdev);
        pipeline_crop_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack4->create(LayerShaderType::crop_pack4, opt, specializations);
    }

    // pack1to4
    if (out_shape.dims == 0 || out_elempack == 4)
    {
        pipeline_crop_pack1to4 = new Pipeline(vkdev);
        pipeline_crop_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack1to4->create(LayerShaderType::crop_pack1to4, opt, specializations);
    }

    // pack4to1
    if (out_shape.dims == 0 || out_elempack == 1)
    {
        pipeline_crop_pack4to1 = new Pipeline(vkdev);
        pipeline_crop_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack4to1->create(LayerShaderType::crop_pack4to1, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || (elempack == 8 && out_elempack == 8))
    {
        pipeline_crop_pack8 = new Pipeline(vkdev);
        pipeline_crop_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack8->create(LayerShaderType::crop_pack8, opt, specializations);
    }

    // pack1to8
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || out_elempack == 8)
    {
        pipeline_crop_pack1to8 = new Pipeline(vkdev);
        pipeline_crop_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack1to8->create(LayerShaderType::crop_pack1to8, opt, specializations);
    }

    // pack4to8
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || out_elempack == 8)
    {
        pipeline_crop_pack4to8 = new Pipeline(vkdev);
        pipeline_crop_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack4to8->create(LayerShaderType::crop_pack4to8, opt, specializations);
    }

    // pack8to4
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || (elempack == 8 && out_elempack == 4))
    {
        pipeline_crop_pack8to4 = new Pipeline(vkdev);
        pipeline_crop_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack8to4->create(LayerShaderType::crop_pack8to4, opt, specializations);
    }

    // pack8to1
    if ((opt.use_shader_pack8 && out_shape.dims == 0) || (elempack == 8 && out_elempack == 1))
    {
        pipeline_crop_pack8to1 = new Pipeline(vkdev);
        pipeline_crop_pack8to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_crop_pack8to1->create(LayerShaderType::crop_pack8to1, opt, specializations);
    }

    return 0;
}

int Crop_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_crop;
    pipeline_crop = 0;

    delete pipeline_crop_pack4;
    pipeline_crop_pack4 = 0;

    delete pipeline_crop_pack1to4;
    pipeline_crop_pack1to4 = 0;

    delete pipeline_crop_pack4to1;
    pipeline_crop_pack4to1 = 0;

    delete pipeline_crop_pack8;
    pipeline_crop_pack8 = 0;

    delete pipeline_crop_pack1to8;
    pipeline_crop_pack1to8 = 0;

    delete pipeline_crop_pack4to8;
    pipeline_crop_pack4to8 = 0;

    delete pipeline_crop_pack8to4;
    pipeline_crop_pack8to4 = 0;

    delete pipeline_crop_pack8to1;
    pipeline_crop_pack8to1 = 0;

    return 0;
}

int Crop_vulkan::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int _woffset, _hoffset, _coffset;
    int _outw, _outh, _outc;
    // resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
    _outw = output_w;
    _outh = output_h;
    _outc = output_c;
    _woffset = offset_w;
    _hoffset = offset_h;
    _coffset = offset_c;

    // TODO vec and image crop

    if (dims == 3)
    {
        if (_woffset == 0 && _hoffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outc == bottom_blob.c * elempack)
        {
            top_blob = bottom_blob;
            return 0;
        }

        int offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8
                                                     : _coffset % 4 == 0                           ? 4
                                                                                                   : 1;

        int out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4
                                                                                       : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        // unpacking
        VkTensor bottom_blob_unpacked = bottom_blob;
        if (elempack == out_elempack && elempack > offset_elempack)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

            vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, offset_elempack, cmd, opt_pack1);
        }

        top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkTensor> bindings(2);
        bindings[0] = bottom_blob_unpacked;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(13);
        constants[0].i = bottom_blob_unpacked.dims;
        constants[1].i = bottom_blob_unpacked.w;
        constants[2].i = bottom_blob_unpacked.h;
        constants[3].i = bottom_blob_unpacked.c;
        constants[4].i = bottom_blob_unpacked.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;
        constants[10].i = _woffset;
        constants[11].i = _hoffset;
        constants[12].i = _coffset;

        const Pipeline* pipeline = 0;
        if (elempack == 1 && out_elempack == 1)
        {
            pipeline = pipeline_crop;
        }
        else if (elempack == 4 && offset_elempack == 4 && out_elempack == 4)
        {
            constants[12].i = _coffset / 4;

            pipeline = pipeline_crop_pack4;
        }
        else if (elempack == 4 && offset_elempack == 1 && out_elempack == 4)
        {
            pipeline = pipeline_crop_pack1to4;
        }
        else if (elempack == 1 && out_elempack == 4)
        {
            pipeline = pipeline_crop_pack1to4;
        }
        else if (elempack == 4 && out_elempack == 1)
        {
            pipeline = pipeline_crop_pack4to1;
        }
        else if (elempack == 8 && offset_elempack == 8 && out_elempack == 8)
        {
            constants[12].i = _coffset / 8;

            pipeline = pipeline_crop_pack8;
        }
        else if (elempack == 8 && offset_elempack == 4 && out_elempack == 8)
        {
            pipeline = pipeline_crop_pack4to8;
        }
        else if (elempack == 8 && offset_elempack == 1 && out_elempack == 8)
        {
            pipeline = pipeline_crop_pack1to8;
        }
        else if (elempack == 1 && out_elempack == 8)
        {
            pipeline = pipeline_crop_pack1to8;
        }
        else if (elempack == 4 && out_elempack == 8)
        {
            pipeline = pipeline_crop_pack4to8;
        }
        else if (elempack == 8 && out_elempack == 4)
        {
            pipeline = pipeline_crop_pack8to4;
        }
        else if (elempack == 8 && out_elempack == 1)
        {
            pipeline = pipeline_crop_pack8to1;
        }
        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
}

int Crop_vulkan::record_pipeline(const std::vector<VkTensor>& bottom_blobs, std::vector<VkTensor>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkTensor& bottom_blob = bottom_blobs[0];
    const VkTensor& reference_blob = bottom_blobs[1];

    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int _woffset, _hoffset, _coffset;
    int _outw, _outh, _outc;
    // if (woffset == -233)
    // {
    //     resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob.mapped(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
    // }
    // else
    // {
    //     resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
    // }
    _outw = output_w;
    _outh = output_h;
    _outc = output_c;
    _woffset = 0; // offset_w;
    _hoffset = 0; // offset_h;
    _coffset = 0; // offset_c;

    // TODO vec and image crop

    if (dims == 3)
    {
        if (_woffset == 0 && _hoffset == 0 && _coffset == 0 && _outw == bottom_blob.w && _outh == bottom_blob.h && _outc == bottom_blob.c * elempack)
        {
            top_blobs[0] = bottom_blob;
            return 0;
        }

        int offset_elempack = _coffset == 0 ? elempack : opt.use_shader_pack8 && _coffset % 8 == 0 ? 8
                                                     : _coffset % 4 == 0                           ? 4
                                                                                                   : 1;

        int out_elempack = opt.use_shader_pack8 && _outc % 8 == 0 ? 8 : _outc % 4 == 0 ? 4
                                                                                       : 1;
        size_t out_elemsize = elemsize / elempack * out_elempack;

        if (opt.use_fp16_packed && !opt.use_fp16_storage)
        {
            if (out_elempack == 8) out_elemsize = 8 * 2u;
            if (out_elempack == 4) out_elemsize = 4 * 2u;
            if (out_elempack == 1) out_elemsize = 4u;
        }

        // unpacking
        VkTensor bottom_blob_unpacked = bottom_blob;
        if (elempack == out_elempack && elempack > offset_elempack)
        {
            Option opt_pack1 = opt;
            opt_pack1.blob_vkallocator = opt.workspace_vkallocator;

            vkdev->convert_packing(bottom_blob, bottom_blob_unpacked, offset_elempack, cmd, opt_pack1);
        }

        VkTensor& top_blob = top_blobs[0];

        top_blob.create(_outw, _outh, _outc / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkTensor> bindings(2);
        bindings[0] = bottom_blob_unpacked;
        bindings[1] = top_blob;

        std::vector<vk_constant_type> constants(13);
        constants[0].i = bottom_blob_unpacked.dims;
        constants[1].i = bottom_blob_unpacked.w;
        constants[2].i = bottom_blob_unpacked.h;
        constants[3].i = bottom_blob_unpacked.c;
        constants[4].i = bottom_blob_unpacked.cstep;
        constants[5].i = top_blob.dims;
        constants[6].i = top_blob.w;
        constants[7].i = top_blob.h;
        constants[8].i = top_blob.c;
        constants[9].i = top_blob.cstep;
        constants[10].i = _woffset;
        constants[11].i = _hoffset;
        constants[12].i = _coffset;

        const Pipeline* pipeline = 0;
        if (elempack == 1 && out_elempack == 1)
        {
            pipeline = pipeline_crop;
        }
        else if (elempack == 4 && offset_elempack == 4 && out_elempack == 4)
        {
            constants[12].i = _coffset / 4;

            pipeline = pipeline_crop_pack4;
        }
        else if (elempack == 4 && offset_elempack == 1 && out_elempack == 4)
        {
            pipeline = pipeline_crop_pack1to4;
        }
        else if (elempack == 1 && out_elempack == 4)
        {
            pipeline = pipeline_crop_pack1to4;
        }
        else if (elempack == 4 && out_elempack == 1)
        {
            pipeline = pipeline_crop_pack4to1;
        }
        else if (elempack == 8 && offset_elempack == 8 && out_elempack == 8)
        {
            constants[12].i = _coffset / 8;

            pipeline = pipeline_crop_pack8;
        }
        else if (elempack == 8 && offset_elempack == 4 && out_elempack == 8)
        {
            pipeline = pipeline_crop_pack4to8;
        }
        else if (elempack == 8 && offset_elempack == 1 && out_elempack == 8)
        {
            pipeline = pipeline_crop_pack1to8;
        }
        else if (elempack == 1 && out_elempack == 8)
        {
            pipeline = pipeline_crop_pack1to8;
        }
        else if (elempack == 4 && out_elempack == 8)
        {
            pipeline = pipeline_crop_pack4to8;
        }
        else if (elempack == 8 && out_elempack == 4)
        {
            pipeline = pipeline_crop_pack8to4;
        }
        else if (elempack == 8 && out_elempack == 1)
        {
            pipeline = pipeline_crop_pack8to1;
        }

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
}

} // namespace TEngine
