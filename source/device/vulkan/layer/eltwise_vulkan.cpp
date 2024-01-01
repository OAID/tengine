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

#include "eltwise_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

Eltwise_vulkan::Eltwise_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = false;
    pipeline_eltwise[0] = 0;
    pipeline_eltwise[1] = 0;
    pipeline_eltwise_pack4[0] = 0;
    pipeline_eltwise_pack4[1] = 0;
    pipeline_eltwise_pack8[0] = 0;
    pipeline_eltwise_pack8[1] = 0;

    graph = ir_graph;
    node = ir_node;

    for (int i = 0; i < ir_node->input_num; i++)
    {
        struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[i]);
        std::string name = input->name;
        bottoms.push_back(name);
    }

    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    std::string name = output->name;
    tops.push_back(name);

	output_c = output->dims[1];
	output_h = output->dims[2];
	output_w = output->dims[3];

    struct eltwise_param* param = (struct eltwise_param*)ir_node->op.param_mem;
    op_type = (param->type) / 2;
}

int Eltwise_vulkan::create_pipeline(const Option& opt)
{
    const Tensor& shape = Tensor(output_w, output_h, output_c, (void*)0); // top_shapes.empty() ? Tensor() : top_shapes[0];

    int elempack = 1;
    if (shape.dims == 1) elempack = opt.use_shader_pack8 && shape.w % 8 == 0 ? 8 : shape.w % 4 == 0 ? 4
                                                                                                    : 1;
    if (shape.dims == 2) elempack = opt.use_shader_pack8 && shape.h % 8 == 0 ? 8 : shape.h % 4 == 0 ? 4
                                                                                                    : 1;
    if (shape.dims == 3) elempack = opt.use_shader_pack8 && shape.c % 8 == 0 ? 8 : shape.c % 4 == 0 ? 4
                                                                                                    : 1;

    size_t elemsize;
    if (opt.use_fp16_storage)
    {
        elemsize = elempack * 2u;
    }
    else if (opt.use_fp16_packed)
    {
        elemsize = elempack == 1 ? 4u : elempack * 2u;
    }
    else
    {
        elemsize = elempack * 4u;
    }

    Tensor shape_packed;
    if (shape.dims == 1) shape_packed = Tensor(shape.w / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 2) shape_packed = Tensor(shape.w, shape.h / elempack, (void*)0, elemsize, elempack);
    if (shape.dims == 3) shape_packed = Tensor(shape.w, shape.h, shape.c / elempack, (void*)0, elemsize, elempack);

    std::vector<vk_specialization_type> specializations(2 + 5);
    specializations[0].i = op_type;
    specializations[1].i = 0;     // coeffs.w == 0 ? 0 : 1;   TODO fix coeffs value
    specializations[2 + 0].i = 0; // shape_packed.dims;
    specializations[2 + 1].i = 0; // shape_packed.w;
    specializations[2 + 2].i = 0; // shape_packed.h;
    specializations[2 + 3].i = 0; // shape_packed.c;
    specializations[2 + 4].i = 0; // shape_packed.cstep;

    Tensor local_size_xyz;
    if (shape_packed.dims == 1)
    {
        local_size_xyz.w = std::min(64, shape_packed.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }
    if (shape_packed.dims == 2)
    {
        local_size_xyz.w = std::min(8, shape_packed.w);
        local_size_xyz.h = std::min(8, shape_packed.h);
        local_size_xyz.c = 1;
    }
    if (shape_packed.dims == 3)
    {
        local_size_xyz.w = std::min(4, shape_packed.w);
        local_size_xyz.h = std::min(4, shape_packed.h);
        local_size_xyz.c = std::min(4, shape_packed.c);
    }

    // pack1
    if (shape.dims == 0 || elempack == 1)
    {
        pipeline_eltwise[0] = new Pipeline(vkdev);
        pipeline_eltwise[0]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_eltwise[0]->create(LayerShaderType::eltwise, opt, specializations);
        pipeline_eltwise[1] = new Pipeline(vkdev);
        pipeline_eltwise[1]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_eltwise[1]->create(LayerShaderType::eltwise, opt, specializations);
    }

    // pack4
    if (shape.dims == 0 || elempack == 4)
    {
        pipeline_eltwise_pack4[0] = new Pipeline(vkdev);
        pipeline_eltwise_pack4[0]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_eltwise_pack4[0]->create(LayerShaderType::eltwise_pack4, opt, specializations);
        pipeline_eltwise_pack4[1] = new Pipeline(vkdev);
        pipeline_eltwise_pack4[1]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_eltwise_pack4[1]->create(LayerShaderType::eltwise_pack4, opt, specializations);
    }

    // pack8
    if ((opt.use_shader_pack8 && shape.dims == 0) || elempack == 8)
    {
        pipeline_eltwise_pack8[0] = new Pipeline(vkdev);
        pipeline_eltwise_pack8[0]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_eltwise_pack8[0]->create(LayerShaderType::eltwise_pack8, opt, specializations);
        pipeline_eltwise_pack8[1] = new Pipeline(vkdev);
        pipeline_eltwise_pack8[1]->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_eltwise_pack8[1]->create(LayerShaderType::eltwise_pack8, opt, specializations);
    }

    return 0;
}

int Eltwise_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_eltwise[0];
    delete pipeline_eltwise[1];
    pipeline_eltwise[0] = 0;
    pipeline_eltwise[1] = 0;

    delete pipeline_eltwise_pack4[0];
    delete pipeline_eltwise_pack4[1];
    pipeline_eltwise_pack4[0] = 0;
    pipeline_eltwise_pack4[1] = 0;

    delete pipeline_eltwise_pack8[0];
    delete pipeline_eltwise_pack8[1];
    pipeline_eltwise_pack8[0] = 0;
    pipeline_eltwise_pack8[1] = 0;

    return 0;
}

int Eltwise_vulkan::record_pipeline(const std::vector<VkTensor>& bottom_blobs, std::vector<VkTensor>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    const VkTensor& bottom_blob = bottom_blobs[0];
    const VkTensor& bottom_blob1 = bottom_blobs[1];

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int elempack = bottom_blob.elempack;

    VkTensor& top_blob = top_blobs[0];
    top_blob.create(w, h, channels, elemsize, elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkTensor> bindings(3);
    bindings[0] = bottom_blob;
    bindings[1] = bottom_blob1;
    bindings[2] = top_blob;

    std::vector<vk_constant_type> constants(5 + 2);
    constants[0].i = top_blob.dims;
    constants[1].i = top_blob.w;
    constants[2].i = top_blob.h;
    constants[3].i = top_blob.c;
    constants[4].i = top_blob.cstep;
    constants[5].f = 1.0f; // coeffs.w == 0 ? 1.f : coeffs[0];     TODO fix coeffs value
    constants[6].f = 1.0f; // coeffs.w == 0 ? 1.f : coeffs[1];

    const Pipeline* pipeline = elempack == 8   ? pipeline_eltwise_pack8[1]
                               : elempack == 4 ? pipeline_eltwise_pack4[1]
                                               : pipeline_eltwise[1];

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    for (size_t b = 2; b < bottom_blobs.size(); b++)
    {
        std::vector<VkTensor> bindings(3);
        bindings[0] = top_blob;
        bindings[1] = bottom_blobs[b];
        bindings[2] = top_blob; // TODO use separated pipeline ?

        std::vector<vk_constant_type> constants(5 + 2);
        constants[0].i = top_blob.dims;
        constants[1].i = top_blob.w;
        constants[2].i = top_blob.h;
        constants[3].i = top_blob.c;
        constants[4].i = top_blob.cstep;
        constants[5].f = 1.f;
        constants[6].f = 1.0f; // coeffs.w == 0 ? 1 : coeffs[b];       TODO fixcoeffs value

        const Pipeline* pipeline = elempack == 8   ? pipeline_eltwise_pack8[b % 2]
                                   : elempack == 4 ? pipeline_eltwise_pack4[b % 2]
                                                   : pipeline_eltwise[b % 2];

        cmd.record_pipeline(pipeline, bindings, constants, top_blob);
    }

    return 0;
}

} // namespace TEngine
