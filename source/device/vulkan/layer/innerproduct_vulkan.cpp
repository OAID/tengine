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

#include "innerproduct_vulkan.hpp"
#include "../layer_shader_type.h"
#include "vulkan_layer.hpp"

namespace TEngine {

InnerProduct_vulkan::InnerProduct_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev)
    : Layer(vkdev)
{
    one_blob_only = true;
    flatten = 0;

    pipeline_innerproduct = 0;
    pipeline_innerproduct_pack4 = 0;
    pipeline_innerproduct_pack1to4 = 0;
    pipeline_innerproduct_pack4to1 = 0;
    pipeline_innerproduct_pack8 = 0;
    pipeline_innerproduct_pack1to8 = 0;
    pipeline_innerproduct_pack4to8 = 0;
    pipeline_innerproduct_pack8to4 = 0;
    pipeline_innerproduct_pack8to1 = 0;

    graph = ir_graph;
    node = ir_node;

    struct tensor* input = get_ir_graph_tensor(graph, node->input_tensors[0]);
    std::string name = input->name;
    bottoms.push_back(name);

    struct tensor* output = get_ir_graph_tensor(graph, node->output_tensors[0]);
    name = output->name;
    tops.push_back(name);

    struct fc_param* param = (struct fc_param*)ir_node->op.param_mem;

    num_output = param->num_output;
    input_c = input->dims[1]; // param->input_channel;
    input_h = input->dims[2];
    input_w = input->dims[3];
    output_c = output->dims[1]; // param->output_channel;
    output_h = output->dims[2];
    output_w = output->dims[3];

    struct tensor* weight = get_ir_graph_tensor(graph, node->input_tensors[1]);
    weight_data_size = weight->elem_num;

    activation_type = -1;
}

int InnerProduct_vulkan::create_pipeline(const Option& _opt)
{
    Option opt = _opt;
    const Tensor& shape = Tensor(input_w, input_h, input_c, (void*)0);        // bottom_shapes.empty() ? Tensor() : bottom_shapes[0];
    const Tensor& out_shape = Tensor(output_w, output_h, output_c, (void*)0); // top_shapes.empty() ? Tensor() : top_shapes[0];

    Tensor shape_flatten;
    if (shape.dims != 0)
    {
        shape_flatten = Tensor(shape.w * shape.h * shape.c, (void*)0);
    }

    int num_input = weight_data_size / num_output;

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4
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

    Tensor shape_flatten_packed;
    if (shape_flatten.dims == 1) shape_flatten_packed = Tensor(shape_flatten.w / elempack, (void*)0, elemsize, elempack);

    Tensor out_shape_packed;
    if (out_shape.dims == 1) out_shape_packed = Tensor(out_shape.w / out_elempack, (void*)0, out_elemsize, out_elempack);

    {
        opt.use_image_storage = false;
    }

    {
        flatten = new Flatten_vulkan(vkdev);

        flatten->input_w = shape.w;
        flatten->input_h = shape.h;
        flatten->input_c = shape.c;
        flatten->output_w = shape_flatten.w;
        flatten->output_h = shape_flatten.h;
        flatten->output_c = shape_flatten.c;
        flatten->output_size = shape_flatten.w * shape_flatten.h * shape_flatten.c;

        flatten->create_pipeline(opt);
    }

    std::vector<vk_specialization_type> specializations(4 + 10);
    specializations[0].i = bias_term;
    specializations[1].i = activation_type;
    specializations[2].f = 0.f;   // activation_params.w >= 1 ? activation_params[0] : 0.f;
    specializations[3].f = 0.f;   // activation_params.w == 2 ? activation_params[1] : 0.f;
    specializations[4 + 0].i = 0; // shape_flatten_packed.dims;
    specializations[4 + 1].i = 0; // shape_flatten_packed.w;
    specializations[4 + 2].i = 0; // shape_flatten_packed.h;
    specializations[4 + 3].i = 0; // shape_flatten_packed.c;
    specializations[4 + 4].i = 0; // shape_flatten_packed.cstep;
    specializations[4 + 5].i = 0; // out_shape_packed.dims;
    specializations[4 + 6].i = 0; // out_shape_packed.w;
    specializations[4 + 7].i = 0; // out_shape_packed.h;
    specializations[4 + 8].i = 0; // out_shape_packed.c;
    specializations[4 + 9].i = 0; // out_shape_packed.cstep;

    Tensor local_size_xyz(std::min(64, num_output / out_elempack), 1, 1, (void*)0);
    if (out_shape_packed.dims != 0)
    {
        local_size_xyz.w = std::min(64, out_shape_packed.w);
        local_size_xyz.h = 1;
        local_size_xyz.c = 1;
    }

    // pack1
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline_innerproduct = new Pipeline(vkdev);
        pipeline_innerproduct->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct->create(LayerShaderType::innerproduct, opt, specializations);
    }

    // pack4
    if (elempack == 4 && out_elempack == 4)
    {
        pipeline_innerproduct_pack4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack4->create(LayerShaderType::innerproduct_pack4, opt, specializations);
    }

    // pack1to4
    if (elempack == 1 && out_elempack == 4)
    {
        pipeline_innerproduct_pack1to4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack1to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack1to4->create(LayerShaderType::innerproduct_pack1to4, opt, specializations);
    }

    // pack4to1
    if (elempack == 4 && out_elempack == 1)
    {
        pipeline_innerproduct_pack4to1 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack4to1->create(LayerShaderType::innerproduct_pack4to1, opt, specializations);
    }

    // pack8
    if (elempack == 8 && out_elempack == 8)
    {
        pipeline_innerproduct_pack8 = new Pipeline(vkdev);
        pipeline_innerproduct_pack8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack8->create(LayerShaderType::innerproduct_pack8, opt, specializations);
    }

    // pack1to8
    if (elempack == 1 && out_elempack == 8)
    {
        pipeline_innerproduct_pack1to8 = new Pipeline(vkdev);
        pipeline_innerproduct_pack1to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack1to8->create(LayerShaderType::innerproduct_pack1to8, opt, specializations);
    }

    // pack4to8
    if (elempack == 4 && out_elempack == 8)
    {
        pipeline_innerproduct_pack4to8 = new Pipeline(vkdev);
        pipeline_innerproduct_pack4to8->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack4to8->create(LayerShaderType::innerproduct_pack4to8, opt, specializations);
    }

    // pack8to4
    if (elempack == 8 && out_elempack == 4)
    {
        pipeline_innerproduct_pack8to4 = new Pipeline(vkdev);
        pipeline_innerproduct_pack8to4->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack8to4->create(LayerShaderType::innerproduct_pack8to4, opt, specializations);
    }

    // pack8to1
    if (elempack == 8 && out_elempack == 1)
    {
        pipeline_innerproduct_pack8to1 = new Pipeline(vkdev);
        pipeline_innerproduct_pack8to1->set_optimal_local_size_xyz(local_size_xyz);
        pipeline_innerproduct_pack8to1->create(LayerShaderType::innerproduct_pack8to1, opt, specializations);
    }

    return 0;
}

int InnerProduct_vulkan::destroy_pipeline(const Option& opt)
{
    if (flatten)
    {
        flatten->destroy_pipeline(opt);
        delete flatten;
        flatten = 0;
    }

    delete pipeline_innerproduct;
    pipeline_innerproduct = 0;

    delete pipeline_innerproduct_pack4;
    pipeline_innerproduct_pack4 = 0;

    delete pipeline_innerproduct_pack1to4;
    pipeline_innerproduct_pack1to4 = 0;

    delete pipeline_innerproduct_pack4to1;
    pipeline_innerproduct_pack4to1 = 0;

    delete pipeline_innerproduct_pack8;
    pipeline_innerproduct_pack8 = 0;

    delete pipeline_innerproduct_pack1to8;
    pipeline_innerproduct_pack1to8 = 0;

    delete pipeline_innerproduct_pack4to8;
    pipeline_innerproduct_pack4to8 = 0;

    delete pipeline_innerproduct_pack8to4;
    pipeline_innerproduct_pack8to4 = 0;

    delete pipeline_innerproduct_pack8to1;
    pipeline_innerproduct_pack8to1 = 0;

    return 0;
}

int InnerProduct_vulkan::upload_model(VkTransfer& cmd, const Option& opt)
{
    int num_input = weight_data_size / num_output;

    int elempack = opt.use_shader_pack8 && num_input % 8 == 0 ? 8 : num_input % 4 == 0 ? 4
                                                                                       : 1;
    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4
                                                                                             : 1;

    // src = inch-outch
    // dst = pa-pb-inch/pa-outch/pb
    tensor* weight_tensor = get_ir_graph_tensor(graph, node->input_tensors[1]);
    Tensor weight_data = Tensor(weight_tensor->elem_num, weight_tensor->data);
    Tensor weight_data_packed;
    {
        Tensor weight_data_r2 = weight_data.reshape(num_input, num_output);

        weight_data_packed.create(num_input / elempack, num_output / out_elempack, (size_t)4 * elempack * out_elempack, elempack * out_elempack);

        for (int q = 0; q + (out_elempack - 1) < num_output; q += out_elempack)
        {
            float* g00 = weight_data_packed.row(q / out_elempack);

            for (int p = 0; p + (elempack - 1) < num_input; p += elempack)
            {
                for (int i = 0; i < out_elempack; i++)
                {
                    const float* k0 = weight_data_r2.row(q + i);
                    k0 += p;

                    for (int j = 0; j < elempack; j++)
                    {
                        g00[0] = k0[j];

                        g00++;
                    }
                }
            }
        }
    }

    {
        cmd.record_upload(weight_data_packed, weight_data_gpu, opt);
    }

    if (bias_term)
    {
        tensor* bias_tensor = get_ir_graph_tensor(graph, node->input_tensors[2]);
        Tensor bias_data = Tensor(bias_tensor->elem_num, bias_tensor->data);
        Tensor bias_data_packed;
        convert_packing(bias_data, bias_data_packed, out_elempack);

        {
            cmd.record_upload(bias_data_packed, bias_data_gpu, opt);
        }
    }
    return 0;
}

int InnerProduct_vulkan::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    // flatten
    VkTensor bottom_blob_flattened = bottom_blob;
    {
        Option opt_flatten = opt;
        opt_flatten.blob_vkallocator = opt.workspace_vkallocator;

        flatten->record_pipeline(bottom_blob, bottom_blob_flattened, cmd, opt_flatten);
    }

    size_t elemsize = bottom_blob_flattened.elemsize;
    int elempack = bottom_blob_flattened.elempack;

    int out_elempack = opt.use_shader_pack8 && num_output % 8 == 0 ? 8 : num_output % 4 == 0 ? 4
                                                                                             : 1;
    size_t out_elemsize = elemsize / elempack * out_elempack;

    if (opt.use_fp16_packed && !opt.use_fp16_storage)
    {
        if (out_elempack == 8) out_elemsize = 8 * 2u;
        if (out_elempack == 4) out_elemsize = 4 * 2u;
        if (out_elempack == 1) out_elemsize = 4u;
    }

    top_blob.create(num_output / out_elempack, out_elemsize, out_elempack, opt.blob_vkallocator);
    if (top_blob.empty())
        return -100;

    std::vector<VkTensor> bindings(4);
    bindings[0] = bottom_blob_flattened;
    bindings[1] = top_blob;
    bindings[2] = weight_data_gpu;
    bindings[3] = bias_data_gpu;

    std::vector<vk_constant_type> constants(10);
    constants[0].i = bottom_blob_flattened.dims;
    constants[1].i = bottom_blob_flattened.w;
    constants[2].i = bottom_blob_flattened.h;
    constants[3].i = bottom_blob_flattened.c;
    constants[4].i = bottom_blob_flattened.cstep;
    constants[5].i = top_blob.dims;
    constants[6].i = top_blob.w;
    constants[7].i = top_blob.h;
    constants[8].i = top_blob.c;
    constants[9].i = top_blob.cstep;

    const Pipeline* pipeline = 0;
    if (elempack == 1 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct;
    }
    else if (elempack == 4 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack4;
    }
    else if (elempack == 1 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack1to4;
    }
    else if (elempack == 4 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct_pack4to1;
    }
    else if (elempack == 8 && out_elempack == 8)
    {
        pipeline = pipeline_innerproduct_pack8;
    }
    else if (elempack == 1 && out_elempack == 8)
    {
        pipeline = pipeline_innerproduct_pack1to8;
    }
    else if (elempack == 4 && out_elempack == 8)
    {
        pipeline = pipeline_innerproduct_pack4to8;
    }
    else if (elempack == 8 && out_elempack == 4)
    {
        pipeline = pipeline_innerproduct_pack8to4;
    }
    else if (elempack == 8 && out_elempack == 1)
    {
        pipeline = pipeline_innerproduct_pack8to1;
    }

    cmd.record_pipeline(pipeline, bindings, constants, top_blob);

    return 0;
}

} // namespace TEngine
