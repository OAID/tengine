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

#include "softmax_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

Softmax_vulkan::Softmax_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = true;
    support_inplace = true;
    pipeline_softmax_reduce_max = 0;
    pipeline_softmax_exp_sub_max = 0;
    pipeline_softmax_reduce_sum = 0;
    pipeline_softmax_div_sum = 0;

    pipeline_softmax_reduce_max_pack4 = 0;
    pipeline_softmax_exp_sub_max_pack4 = 0;
    pipeline_softmax_reduce_sum_pack4 = 0;
    pipeline_softmax_div_sum_pack4 = 0;

    pipeline_softmax_reduce_max_pack8 = 0;
    pipeline_softmax_exp_sub_max_pack8 = 0;
    pipeline_softmax_reduce_sum_pack8 = 0;
    pipeline_softmax_div_sum_pack8 = 0;

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

    struct softmax_param* param = (struct softmax_param*)ir_node->op.param_mem;
    axis = param->axis - 1;
}

int Softmax_vulkan::create_pipeline(const Option& opt)
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

    Tensor workspace_shape_packed;
    if (shape.dims == 1) // axis == 0
    {
        workspace_shape_packed = Tensor(1, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 2 && axis == 0)
    {
        workspace_shape_packed = Tensor(shape.w, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 2 && axis == 1)
    {
        workspace_shape_packed = Tensor(shape.h / elempack, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 3 && axis == 0)
    {
        workspace_shape_packed = Tensor(shape.w, shape.h, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 3 && axis == 1)
    {
        workspace_shape_packed = Tensor(shape.w, shape.c / elempack, (void*)0, elemsize, elempack);
    }
    else if (shape.dims == 3 && axis == 2)
    {
        workspace_shape_packed = Tensor(shape.h, shape.c / elempack, (void*)0, elemsize, elempack);
    }

    std::vector<vk_specialization_type> specializations(1 + 10);
    specializations[0].i = axis;
    specializations[1 + 0].i = 0; // shape_packed.dims;
    specializations[1 + 1].i = 0; // shape_packed.w;
    specializations[1 + 2].i = 0; // shape_packed.h;
    specializations[1 + 3].i = 0; // shape_packed.c;
    specializations[1 + 4].i = 0; // shape_packed.cstep;
    specializations[1 + 5].i = 0; // workspace_shape_packed.dims;
    specializations[1 + 6].i = 0; // workspace_shape_packed.w;
    specializations[1 + 7].i = 0; // workspace_shape_packed.h;
    specializations[1 + 8].i = 0; // workspace_shape_packed.c;
    specializations[1 + 9].i = 0; // workspace_shape_packed.cstep;

    {
        Tensor local_size_xyz;
        if (workspace_shape_packed.dims == 1)
        {
            local_size_xyz.w = std::min(64, workspace_shape_packed.w);
            local_size_xyz.h = 1;
            local_size_xyz.c = 1;
        }
        if (workspace_shape_packed.dims == 2)
        {
            local_size_xyz.w = std::min(8, workspace_shape_packed.w);
            local_size_xyz.h = std::min(8, workspace_shape_packed.h);
            local_size_xyz.c = 1;
        }
        if (workspace_shape_packed.dims != 0)
        {
            local_size_xyz.w = std::min(4, workspace_shape_packed.w);
            local_size_xyz.h = std::min(4, workspace_shape_packed.h);
            local_size_xyz.c = std::min(4, workspace_shape_packed.c);
        }

        // pack1
        {
            pipeline_softmax_reduce_max = new Pipeline(vkdev);
            pipeline_softmax_reduce_sum = new Pipeline(vkdev);

            pipeline_softmax_reduce_max->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_reduce_sum->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_reduce_max->create(LayerShaderType::softmax_reduce_max, opt, specializations);
            pipeline_softmax_reduce_sum->create(LayerShaderType::softmax_reduce_sum, opt, specializations);
        }

        // pack4
        {
            pipeline_softmax_reduce_max_pack4 = new Pipeline(vkdev);
            pipeline_softmax_reduce_sum_pack4 = new Pipeline(vkdev);

            pipeline_softmax_reduce_max_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_reduce_sum_pack4->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_reduce_max_pack4->create(LayerShaderType::softmax_reduce_max_pack4, opt, specializations);
            pipeline_softmax_reduce_sum_pack4->create(LayerShaderType::softmax_reduce_sum_pack4, opt, specializations);
        }

        // pack8
        if (opt.use_shader_pack8)
        {
            pipeline_softmax_reduce_max_pack8 = new Pipeline(vkdev);
            pipeline_softmax_reduce_sum_pack8 = new Pipeline(vkdev);

            pipeline_softmax_reduce_max_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_reduce_sum_pack8->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_reduce_max_pack8->create(LayerShaderType::softmax_reduce_max_pack8, opt, specializations);
            pipeline_softmax_reduce_sum_pack8->create(LayerShaderType::softmax_reduce_sum_pack8, opt, specializations);
        }
    }

    {
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
        {
            pipeline_softmax_exp_sub_max = new Pipeline(vkdev);
            pipeline_softmax_div_sum = new Pipeline(vkdev);

            pipeline_softmax_exp_sub_max->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_div_sum->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_exp_sub_max->create(LayerShaderType::softmax_exp_sub_max, opt, specializations);
            pipeline_softmax_div_sum->create(LayerShaderType::softmax_div_sum, opt, specializations);
        }

        // pack4
        {
            pipeline_softmax_exp_sub_max_pack4 = new Pipeline(vkdev);
            pipeline_softmax_div_sum_pack4 = new Pipeline(vkdev);

            pipeline_softmax_exp_sub_max_pack4->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_div_sum_pack4->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_exp_sub_max_pack4->create(LayerShaderType::softmax_exp_sub_max_pack4, opt, specializations);
            pipeline_softmax_div_sum_pack4->create(LayerShaderType::softmax_div_sum_pack4, opt, specializations);
        }

        // pack8
        if (opt.use_shader_pack8)
        {
            pipeline_softmax_exp_sub_max_pack8 = new Pipeline(vkdev);
            pipeline_softmax_div_sum_pack8 = new Pipeline(vkdev);

            pipeline_softmax_exp_sub_max_pack8->set_optimal_local_size_xyz(local_size_xyz);
            pipeline_softmax_div_sum_pack8->set_optimal_local_size_xyz(local_size_xyz);

            pipeline_softmax_exp_sub_max_pack8->create(LayerShaderType::softmax_exp_sub_max_pack8, opt, specializations);
            pipeline_softmax_div_sum_pack8->create(LayerShaderType::softmax_div_sum_pack8, opt, specializations);
        }
    }

    return 0;
}

int Softmax_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_softmax_reduce_max;
    pipeline_softmax_reduce_max = 0;

    delete pipeline_softmax_exp_sub_max;
    pipeline_softmax_exp_sub_max = 0;

    delete pipeline_softmax_reduce_sum;
    pipeline_softmax_reduce_sum = 0;

    delete pipeline_softmax_div_sum;
    pipeline_softmax_div_sum = 0;

    delete pipeline_softmax_reduce_max_pack4;
    pipeline_softmax_reduce_max_pack4 = 0;

    delete pipeline_softmax_exp_sub_max_pack4;
    pipeline_softmax_exp_sub_max_pack4 = 0;

    delete pipeline_softmax_reduce_sum_pack4;
    pipeline_softmax_reduce_sum_pack4 = 0;

    delete pipeline_softmax_div_sum_pack4;
    pipeline_softmax_div_sum_pack4 = 0;

    delete pipeline_softmax_reduce_max_pack8;
    pipeline_softmax_reduce_max_pack8 = 0;

    delete pipeline_softmax_exp_sub_max_pack8;
    pipeline_softmax_exp_sub_max_pack8 = 0;

    delete pipeline_softmax_reduce_sum_pack8;
    pipeline_softmax_reduce_sum_pack8 = 0;

    delete pipeline_softmax_div_sum_pack8;
    pipeline_softmax_div_sum_pack8 = 0;

    return 0;
}

int Softmax_vulkan::record_pipeline(VkTensor& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int channels = bottom_top_blob.c;
    size_t elemsize = bottom_top_blob.elemsize;
    int elempack = bottom_top_blob.elempack;

    VkTensor max_workspace;
    VkTensor sum_workspace;

    if (dims == 1) // axis == 0
    {
        max_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(1, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 2 && axis == 0)
    {
        max_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 2 && axis == 1)
    {
        max_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(h, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && axis == 0)
    {
        max_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, h, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && axis == 1)
    {
        max_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(w, channels, elemsize, elempack, opt.workspace_vkallocator);
    }
    else if (dims == 3 && axis == 2)
    {
        max_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator);
        sum_workspace.create(h, channels, elemsize, elempack, opt.workspace_vkallocator);
    }

    // reduce max
    {
        std::vector<VkTensor> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = max_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = max_workspace.dims;
        constants[6].i = max_workspace.w;
        constants[7].i = max_workspace.h;
        constants[8].i = max_workspace.c;
        constants[9].i = max_workspace.cstep;

        const Pipeline* pipeline = elempack == 8   ? pipeline_softmax_reduce_max_pack8
                                   : elempack == 4 ? pipeline_softmax_reduce_max_pack4
                                                   : pipeline_softmax_reduce_max;

        cmd.record_pipeline(pipeline, bindings, constants, max_workspace);
    }

    // exp( v - max )
    {
        std::vector<VkTensor> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = max_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = max_workspace.dims;
        constants[6].i = max_workspace.w;
        constants[7].i = max_workspace.h;
        constants[8].i = max_workspace.c;
        constants[9].i = max_workspace.cstep;

        const Pipeline* pipeline = elempack == 8   ? pipeline_softmax_exp_sub_max_pack8
                                   : elempack == 4 ? pipeline_softmax_exp_sub_max_pack4
                                                   : pipeline_softmax_exp_sub_max;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    // reduce sum
    {
        std::vector<VkTensor> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = sum_workspace.dims;
        constants[6].i = sum_workspace.w;
        constants[7].i = sum_workspace.h;
        constants[8].i = sum_workspace.c;
        constants[9].i = sum_workspace.cstep;

        const Pipeline* pipeline = elempack == 8   ? pipeline_softmax_reduce_sum_pack8
                                   : elempack == 4 ? pipeline_softmax_reduce_sum_pack4
                                                   : pipeline_softmax_reduce_sum;

        cmd.record_pipeline(pipeline, bindings, constants, sum_workspace);
    }

    // div sum
    {
        std::vector<VkTensor> bindings(2);
        bindings[0] = bottom_top_blob;
        bindings[1] = sum_workspace;

        std::vector<vk_constant_type> constants(10);
        constants[0].i = bottom_top_blob.dims;
        constants[1].i = bottom_top_blob.w;
        constants[2].i = bottom_top_blob.h;
        constants[3].i = bottom_top_blob.c;
        constants[4].i = bottom_top_blob.cstep;
        constants[5].i = sum_workspace.dims;
        constants[6].i = sum_workspace.w;
        constants[7].i = sum_workspace.h;
        constants[8].i = sum_workspace.c;
        constants[9].i = sum_workspace.cstep;

        const Pipeline* pipeline = elempack == 8   ? pipeline_softmax_div_sum_pack8
                                   : elempack == 4 ? pipeline_softmax_div_sum_pack4
                                                   : pipeline_softmax_div_sum;

        cmd.record_pipeline(pipeline, bindings, constants, bottom_top_blob);
    }

    return 0;
}

} // namespace TEngine
