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

#include "interp_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

Interp_vulkan::Interp_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = true;
    pipeline_interp = 0;
    pipeline_interp_pack4 = 0;
    pipeline_interp_pack8 = 0;

    pipeline_interp_bicubic_coeffs_x = 0;
    pipeline_interp_bicubic_coeffs_y = 0;
    pipeline_interp_bicubic = 0;
    pipeline_interp_bicubic_pack4 = 0;
    pipeline_interp_bicubic_pack8 = 0;

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
    output_c = output->dims[1]; // param->output_channel;
    output_h = output->dims[2];
    output_w = output->dims[3];

    struct interp_param* param = (struct interp_param*)ir_node->op.param_mem;

    if (param->height_scale != 0 && param->width_scale != 0)
    {
        output_height = input_h * param->height_scale;
        output_width = input_w * param->width_scale;
    }
    else
    {
        height_scale = (float)output->dims[2] / (float)input_h;
        width_scale = (float)output->dims[2] / (float)input_w;
    }
    resize_type = 2; //param->resize_type;
}

int Interp_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Tensor& shape = Tensor(input_w, input_h, input_c, (void*)0);        // bottom_shapes.empty() ? Mat() : bottom_shapes[0];
    const Tensor& out_shape = Tensor(output_w, output_h, output_c, (void*)0); // top_shapes.empty() ? Mat() : top_shapes[0];

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

    // check blob shape
    {
        opt.use_image_storage = false;
    }

    if (resize_type == 1 || resize_type == 2)
    {
        std::vector<vk_specialization_type> specializations(1 + 10);
        specializations[0].i = resize_type;
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

        Tensor local_size_xyz;
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
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_interp = new Pipeline(vkdev);
            pipeline_interp->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp->create(LayerShaderType::interp, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_interp_pack4 = new Pipeline(vkdev);
            pipeline_interp_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_pack4->create(LayerShaderType::interp_pack4, opt, specializations);
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_interp_pack8 = new Pipeline(vkdev);
            pipeline_interp_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_pack8->create(LayerShaderType::interp_pack8, opt, specializations);
        }
    }

    if (resize_type == 3)
    {
        {
            std::vector<vk_specialization_type> specializations(0 + 2);
            specializations[0 + 0].i = shape_packed.w;
            specializations[0 + 1].i = out_shape_packed.w;

            Tensor local_size_xyz(64, 1, 1, (void*)0);
            if (out_shape_packed.dims != 0)
            {
                local_size_xyz.w = std::min(64, out_shape_packed.w);
                local_size_xyz.h = 1;
                local_size_xyz.c = 1;
            }

            pipeline_interp_bicubic_coeffs_x = new Pipeline(vkdev);
            pipeline_interp_bicubic_coeffs_x->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic_coeffs_x->create(LayerShaderType::interp_bicubic_coeffs, opt, specializations);
        }
        {
            std::vector<vk_specialization_type> specializations(0 + 2);
            specializations[0 + 0].i = shape_packed.h;
            specializations[0 + 1].i = out_shape_packed.h;

            Tensor local_size_xyz(64, 1, 1, (void*)0);
            if (out_shape_packed.dims != 0)
            {
                local_size_xyz.w = std::min(64, out_shape_packed.h);
                local_size_xyz.h = 1;
                local_size_xyz.c = 1;
            }

            pipeline_interp_bicubic_coeffs_y = new Pipeline(vkdev);
            pipeline_interp_bicubic_coeffs_y->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic_coeffs_y->create(LayerShaderType::interp_bicubic_coeffs, opt, specializations);
        }

        std::vector<vk_specialization_type> specializations(0 + 10);
        specializations[0 + 0].i = 0; // shape_packed.dims;
        specializations[0 + 1].i = 0; // shape_packed.w;
        specializations[0 + 2].i = 0; // shape_packed.h;
        specializations[0 + 3].i = 0; // shape_packed.c;
        specializations[0 + 4].i = 0; // shape_packed.cstep;
        specializations[0 + 5].i = 0; // out_shape_packed.dims;
        specializations[0 + 6].i = 0; // out_shape_packed.w;
        specializations[0 + 7].i = 0; // out_shape_packed.h;
        specializations[0 + 8].i = 0; // out_shape_packed.c;
        specializations[0 + 9].i = 0; // out_shape_packed.cstep;

        Tensor local_size_xyz;
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
        if (shape.dims == 0 || elempack == 1)
        {
            pipeline_interp_bicubic = new Pipeline(vkdev);
            pipeline_interp_bicubic->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic->create(LayerShaderType::interp_bicubic, opt, specializations);
        }

        // pack4
        if (shape.dims == 0 || elempack == 4)
        {
            pipeline_interp_bicubic_pack4 = new Pipeline(vkdev);
            pipeline_interp_bicubic_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic_pack4->create(LayerShaderType::interp_bicubic_pack4, opt, specializations);
        }

        // pack8
        if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
        {
            pipeline_interp_bicubic_pack8 = new Pipeline(vkdev);
            pipeline_interp_bicubic_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_interp_bicubic_pack8->create(LayerShaderType::interp_bicubic_pack8, opt, specializations);
        }
    }

    return 0;
}

int Interp_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_interp;
    pipeline_interp = 0;

    delete pipeline_interp_pack4;
    pipeline_interp_pack4 = 0;

    delete pipeline_interp_pack8;
    pipeline_interp_pack8 = 0;

    delete pipeline_interp_bicubic_coeffs_x;
    pipeline_interp_bicubic_coeffs_x = 0;

    delete pipeline_interp_bicubic_coeffs_y;
    pipeline_interp_bicubic_coeffs_y = 0;

    delete pipeline_interp_bicubic;
    pipeline_interp_bicubic = 0;

    delete pipeline_interp_bicubic_pack4;
    pipeline_interp_bicubic_pack4 = 0;

    delete pipeline_interp_bicubic_pack8;
    pipeline_interp_bicubic_pack8 = 0;

    return 0;
}

int Interp_vulkan::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    int outw = output_width;
    int outh = output_height;
    if (outw == 0 || outh == 0)
    {
        outw = w * width_scale;
        outh = h * height_scale;
    }

    if (outh == h && outw == w)
    {
        top_blob = bottom_blob;
        return 0;
    }

    top_blob.create(outw, outh, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    if (resize_type == 1 || resize_type == 2) // nearest or bilinear
    {
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
        constants[10].f = w / (float)outw;
        constants[11].f = h / (float)outh;

        const Pipeline* pipeline = elempack == 8   ? pipeline_interp_pack8
                                   : elempack == 4 ? pipeline_interp_pack4
                                                   : pipeline_interp;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    else if (resize_type == 3) // bicubic
    {
        VkTensor alpha(outw, (size_t)(elemsize / elempack * 4), 4, opt.workspace_vkallocator);
        if (alpha.empty())
            return -100;

        VkTensor xofs(outw, (size_t)4u, 1, opt.workspace_vkallocator);
        if (xofs.empty())
            return -100;

        {
            std::vector<VkTensor> bindings(2);
            bindings[0] = alpha;
            bindings[1] = xofs;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = bottom_blob.w;
            constants[1].i = outw;
            constants[2].f = (float)bottom_blob.w / outw;

            // record
            cmd.record_pipeline(pipeline_interp_bicubic_coeffs_x, bindings, constants, alpha);
        }

        VkTensor beta(outh, (size_t)(elemsize / elempack * 4), 4, opt.workspace_vkallocator);
        if (beta.empty())
            return -100;

        VkTensor yofs(outh, (size_t)4u, 1, opt.workspace_vkallocator);
        if (yofs.empty())
            return -100;

        {
            std::vector<VkTensor> bindings(2);
            bindings[0] = beta;
            bindings[1] = yofs;

            std::vector<vk_constant_type> constants(3);
            constants[0].i = bottom_blob.h;
            constants[1].i = outh;
            constants[2].f = (float)bottom_blob.h / outh;

            // record
            cmd.record_pipeline(pipeline_interp_bicubic_coeffs_y, bindings, constants, beta);
        }

        std::vector<VkTensor> bindings(6);
        bindings[0] = bottom_blob;
        bindings[1] = top_blob;
        bindings[2] = alpha;
        bindings[3] = xofs;
        bindings[4] = beta;
        bindings[5] = yofs;

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

        const Pipeline* pipeline = elempack == 8   ? pipeline_interp_bicubic_pack8
                                   : elempack == 4 ? pipeline_interp_bicubic_pack4
                                                   : pipeline_interp_bicubic;

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
}

} // namespace TEngine
