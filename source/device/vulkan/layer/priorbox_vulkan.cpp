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

#include "priorbox_vulkan.hpp"
#include "../layer_shader_type.h"

namespace TEngine {

PriorBox_vulkan::PriorBox_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = false;
    pipeline_priorbox = 0;
    pipeline_priorbox_mxnet = 0;

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
    struct tensor* featmap_tensor = get_ir_graph_tensor(graph, node->input_tensors[0]);
    struct tensor* data_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    struct tensor* output_tensor = get_ir_graph_tensor(graph, node->output_tensors[0]);
    input_c = data_tensor->dims[1]; // param->input_channel;
    input_h = data_tensor->dims[2];
    input_w = data_tensor->dims[3];
    output_c = output_tensor->dims[1]; // param->output_channel;
    output_h = output_tensor->dims[2];
    output_w = output_tensor->dims[3];

    const int data_height = data_tensor->dims[2];
    const int data_width = data_tensor->dims[3];
    const int feat_height = featmap_tensor->dims[2];
    const int feat_width = featmap_tensor->dims[3];

    struct priorbox_param* param = (struct priorbox_param*)ir_node->op.param_mem;

    variances[0] = (param->variance)[0];
    variances[1] = (param->variance)[1];
    variances[2] = (param->variance)[2];
    variances[3] = (param->variance)[3];
    flip = param->flip;
    clip = param->clip;

    if (param->image_h == 0 || param->image_w == 0)
    {
        image_width = data_width;
        image_height = data_height;
    }
    else
    {
        image_width = param->image_w;
        image_height = param->image_h;
    }

    if (param->step_h == 0 || param->step_w == 0)
    {
        step_width = (float)(image_width) / feat_width;
        step_height = (float)(image_height) / feat_height;
    }
    else
    {
        step_width = param->step_w;
        step_height = param->step_h;
    }
    int num_priors = param->num_priors;

    offset = param->offset;
    step_mmdetection = 0;   // TODO fix step_mmdetection value
    center_mmdetection = 0; // TODO fix center_mmdetection value

    min_sizes = Tensor(param->min_size_num, param->min_size);
    max_sizes = Tensor(param->max_size_num, param->max_size);
    aspect_ratios = Tensor(param->aspect_ratio_size, param->aspect_ratio);
    TLOG_INFO("size min max aspect:%d %d %d\n", param->min_size_num, param->max_size_num, param->aspect_ratio_size);
}

int PriorBox_vulkan::create_pipeline(const Option& opt)
{
    const Tensor& shape = Tensor(input_w, input_h, input_c, (void*)0); // bottom_shapes.empty() ? Tensor() : bottom_shapes[0];

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

    // caffe style
    {
        int num_min_size = min_sizes.w;
        int num_max_size = max_sizes.w;
        int num_aspect_ratio = aspect_ratios.w;

        int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
        if (flip)
            num_prior += num_min_size * num_aspect_ratio;

        std::vector<vk_specialization_type> specializations(11 + 2);
        specializations[0].i = flip;
        specializations[1].i = clip;
        specializations[2].f = offset;
        specializations[3].f = variances[0];
        specializations[4].f = variances[1];
        specializations[5].f = variances[2];
        specializations[6].f = variances[3];
        specializations[7].i = num_min_size;
        specializations[8].i = num_max_size;
        specializations[9].i = num_aspect_ratio;
        specializations[10].i = num_prior;
        specializations[11 + 0].i = 0; //shape_packed.w;
        specializations[11 + 1].i = 0; //shape_packed.h;

        pipeline_priorbox = new Pipeline(vkdev);
        pipeline_priorbox->set_optimal_local_size_xyz();
        pipeline_priorbox->create(LayerShaderType::priorbox, opt, specializations);
    }

    // mxnet style
    {
        int num_sizes = min_sizes.w;
        int num_ratios = aspect_ratios.w;

        int num_prior = num_sizes - 1 + num_ratios;

        std::vector<vk_specialization_type> specializations(5 + 2);
        specializations[0].i = clip;
        specializations[1].f = offset;
        specializations[2].i = num_sizes;
        specializations[3].i = num_ratios;
        specializations[4].i = num_prior;
        specializations[5 + 0].i = shape_packed.w;
        specializations[5 + 1].i = shape_packed.h;

        pipeline_priorbox_mxnet = new Pipeline(vkdev);
        pipeline_priorbox_mxnet->set_optimal_local_size_xyz();
        pipeline_priorbox_mxnet->create(LayerShaderType::priorbox_mxnet, opt, specializations);
    }

    return 0;
}

int PriorBox_vulkan::destroy_pipeline(const Option& /*opt*/)
{
    delete pipeline_priorbox;
    pipeline_priorbox = 0;

    delete pipeline_priorbox_mxnet;
    pipeline_priorbox_mxnet = 0;

    return 0;
}

int PriorBox_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    cmd.record_upload(min_sizes, min_sizes_gpu, opt);

    if (max_sizes.w > 0)
        cmd.record_upload(max_sizes, max_sizes_gpu, opt);

    cmd.record_upload(aspect_ratios, aspect_ratios_gpu, opt);

    return 0;
}

int PriorBox_vulkan::record_pipeline(const std::vector<VkTensor>& bottom_blobs, std::vector<VkTensor>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;

    if (bottom_blobs.size() == 1 && image_width == -233 && image_height == -233 && max_sizes.empty())
    {
        // mxnet style _contrib_MultiBoxPrior
        float step_w = step_width;
        float step_h = step_height;
        if (step_w == -233)
            step_w = 1.f / (float)w;
        if (step_h == -233)
            step_h = 1.f / (float)h;

        int num_sizes = min_sizes.w;
        int num_ratios = aspect_ratios.w;

        int num_prior = num_sizes - 1 + num_ratios;

        int elempack = 4;

        size_t elemsize = elempack * 4u;
        if (opt.use_fp16_packed || opt.use_fp16_storage)
        {
            elemsize = elempack * 2u;
        }

        VkTensor& top_blob = top_blobs[0];
        top_blob.create(4 * w * h * num_prior / elempack, elemsize, elempack, opt.blob_vkallocator);
        if (top_blob.empty())
            return -100;

        std::vector<VkTensor> bindings(3);
        bindings[0] = top_blob;
        bindings[1] = min_sizes_gpu;
        bindings[2] = aspect_ratios_gpu;

        std::vector<vk_constant_type> constants(4);
        constants[0].i = w;
        constants[1].i = h;
        constants[2].f = step_w;
        constants[3].f = step_h;

        VkTensor dispatcher;
        dispatcher.w = num_sizes;
        dispatcher.h = w;
        dispatcher.c = h;

        cmd.record_pipeline(pipeline_priorbox_mxnet, bindings, constants, dispatcher);

        return 0;
    }

    int image_w = image_width;
    int image_h = image_height;
    if (image_w == -233)
        image_w = bottom_blobs[1].w;
    if (image_h == -233)
        image_h = bottom_blobs[1].h;

    float step_w = step_width;
    float step_h = step_height;
    if (step_w == -233)
        step_w = (float)image_w / w;
    if (step_h == -233)
        step_h = (float)image_h / h;

    int num_min_size = min_sizes.w;
    int num_max_size = max_sizes.w;
    int num_aspect_ratio = aspect_ratios.w;

    int num_prior = num_min_size * num_aspect_ratio + num_min_size + num_max_size;
    if (flip)
        num_prior += num_min_size * num_aspect_ratio;

    size_t elemsize = 4u;
    if (opt.use_fp16_storage)
    {
        elemsize = 2u;
    }

    VkTensor& top_blob = top_blobs[0];
    top_blob.create(4 * w * h * num_prior, 2, elemsize, 1, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkTensor> bindings(4);
    bindings[0] = top_blob;
    bindings[1] = min_sizes_gpu;
    bindings[2] = num_max_size > 0 ? max_sizes_gpu : min_sizes_gpu;
    bindings[3] = aspect_ratios_gpu;

    std::vector<vk_constant_type> constants(6);
    constants[0].i = w;
    constants[1].i = h;
    constants[2].f = image_w;
    constants[3].f = image_h;
    constants[4].f = step_w;
    constants[5].f = step_h;

    VkTensor dispatcher;
    dispatcher.w = num_min_size;
    dispatcher.h = w;
    dispatcher.c = h;

    cmd.record_pipeline(pipeline_priorbox, bindings, constants, dispatcher);

    return 0;
}

} // namespace TEngine
