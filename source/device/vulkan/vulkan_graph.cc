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
 * Copyright (c) 2021, OPEN AI LAB
 * Author: hhchen@openailab.com
 */

#include "vulkan_graph.hpp"
#include "api/c_api.h"
#include "vulkan_executor.hpp"

#include <cstdio>
#include <iostream>
#include "vulkan_graph.hpp"
#include "vulkan_pipeline.hpp"
#include "vulkan_gpu.hpp"
#include "vulkan_command.hpp"
#include "vulkan_allocator.hpp"
#include "vulkan_tensor.hpp"
#include "vulkan_layer.hpp"

#include "layer/convolution_vulkan.hpp"
#include "layer/pooling_vulkan.hpp"
#include "layer/convolutiondepthwise_vulkan.hpp"
#include "layer/innerproduct_vulkan.hpp"
#include "layer/flatten_vulkan.hpp"
#include "layer/softmax_vulkan.hpp"
#include "layer/relu_vulkan.hpp"
#include "layer/dropout_vulkan.hpp"
#include "layer/eltwise_vulkan.hpp"
#include "layer/priorbox_vulkan.hpp"
#include "layer/permute_vulkan.hpp"
#include "layer/concat_vulkan.hpp"
#include "layer/reshape_vulkan.hpp"
#include "layer/interp_vulkan.hpp"
#include "layer/crop_vulkan.hpp"

#include <sys/time.h>
#include <vector>

extern "C" {
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
}

int vulkan_dev_init(struct device* dev)
{
    (void)dev;
    return 0;
}

int vulkan_dev_prerun(struct device* dev, struct subgraph* subgraph, void* options)
{
    subgraph->device_graph = new VULKANEngine;
    auto engine = (VULKANEngine*)subgraph->device_graph;

    return engine->VULKANEnginePreRun(subgraph);
}

int vulkan_dev_run(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (VULKANEngine*)subgraph->device_graph;
    return engine->VULKANEngineRun(subgraph);
}

int vulkan_dev_postrun(struct device* dev, struct subgraph* subgraph)
{
    auto engine = (VULKANEngine*)subgraph->device_graph;
    engine->VULKANEnginePostRun();
    // delete engine;

    return 0;
}

int vulkan_dev_release(struct device* dev)
{
    (void)dev;
    return 0;
}

namespace TEngine {

static double get_cur_time(void)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + (tv.tv_usec / 1000.0);
}

VulkanGraph::VulkanGraph(struct subgraph* graph)
{
    vkdev = get_gpu_device();
    weight_vkallocator = 0;
    weight_staging_vkallocator = 0;

    // set graph options
    if (!vkdev->info.support_fp16_packed || !vkdev->info.support_fp16_storage)
        opt.use_fp16_packed = false;
    if (!vkdev->info.support_fp16_storage)
    {
        opt.use_fp16_storage = false;
        opt.use_shader_pack8 = false;
    }

    if (!vkdev->info.support_fp16_arithmetic)
        opt.use_fp16_arithmetic = false;

    TLOG_INFO("use_fp16_packed %d\n", opt.use_fp16_packed);
    TLOG_INFO("use_fp16_storage %d\n", opt.use_fp16_storage);
    TLOG_INFO("use_shader_pack8 %d\n", opt.use_shader_pack8);
    TLOG_INFO("use_fp16_arithmetic %d\n", opt.use_fp16_arithmetic);

    struct subgraph* subgraph = (struct subgraph*)graph;
    struct graph* ir_graph = subgraph->graph;
    int node_num = subgraph->node_num;

    sgraph = graph;
    for (int i = 0; i < node_num; i++)
    {
        struct node* ir_node = get_ir_graph_node(ir_graph, subgraph->node_list[i]);

        if (ir_node->op.type == OP_CONST || ir_node->op.type == OP_INPUT)
            continue;
        else if (ir_node->op.type == OP_CLIP)
            ir_node->op.type = OP_RELU6;

        if (ir_node->op.type == OP_CONV)
        {
            struct conv_param* conv_param = (struct conv_param*)ir_node->op.param_mem;

            if (conv_param->group == conv_param->output_channel && conv_param->group != 1 && ir_graph->graph_layout == TENGINE_LAYOUT_NCHW) // DW
            {
                Layer* layer = new ConvolutionDepthWise_vulkan(ir_graph, ir_node, vkdev);
                layer->vkdev = vkdev;
                layers.push_back(layer);
            }
            else
            {
                Layer* layer = new Convolution_vulkan(ir_graph, ir_node, vkdev);
                layer->vkdev = vkdev;
                layers.push_back(layer);
            }
        }

        if (ir_node->op.type == OP_POOL)
        {
            Layer* layer = new Pooling_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_FC)
        {
            Layer* layer = new InnerProduct_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_FLATTEN)
        {
            Layer* layer = new Flatten_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_SOFTMAX)
        {
            Layer* layer = new Softmax_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_RELU)
        {
            Layer* layer = new ReLU_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_DROPOUT)
        {
            Layer* layer = new Dropout_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_ELTWISE)
        {
            Layer* layer = new Eltwise_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_PRIORBOX)
        {
            Layer* layer = new PriorBox_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_PERMUTE)
        {
            Layer* layer = new Permute_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_CONCAT)
        {
            Layer* layer = new Concat_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_RESHAPE)
        {
            Layer* layer = new Reshape_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_INTERP || ir_node->op.type == OP_UPSAMPLE)
        {
            Layer* layer = new Interp_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        if (ir_node->op.type == OP_CROP)
        {
            Layer* layer = new Crop_vulkan(ir_graph, ir_node, vkdev);
            layer->vkdev = vkdev;
            layers.push_back(layer);
        }

        for (int i = 0; i < ir_node->input_num; ++i)
        {
            struct tensor* input = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);
            const auto name = input->name;
            tensor_map_[name] = input;
            tensor_map[name] = Tensor(input);
            VkTensor vktensor;
            vktensor_map_[name] = vktensor;
        }

        for (int i = 0; i < ir_node->output_num; ++i)
        {
            struct tensor* output = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[i]);
            const auto name = output->name;
            tensor_map_[name] = output;
            tensor_map[name] = Tensor(output);
        }
    }
}

VulkanGraph::~VulkanGraph()
{
    for (auto& ptr : mem_buf_vector_)
        std::free(ptr);
}

int VulkanGraph::upload_model()
{
    TEngine::VkTransfer cmd(vkdev);
    if (!weight_vkallocator)
    {
        weight_vkallocator = new VkWeightAllocator(vkdev);
    }
    if (!weight_staging_vkallocator)
    {
        weight_staging_vkallocator = new VkWeightStagingAllocator(vkdev);
    }

    Option opt_upload = opt;
    opt_upload.blob_vkallocator = weight_vkallocator;
    opt_upload.workspace_vkallocator = weight_vkallocator;
    opt_upload.staging_vkallocator = weight_staging_vkallocator;

    int layer_size = layers.size();
    for (int i = 0; i < layer_size; i++)
    {
        layers[i]->upload_model(cmd, opt_upload);
    }

    cmd.submit_and_wait();
    return 0;
}

int VulkanGraph::create_pipeline()
{
    for (size_t i = 0; i < layers.size(); i++)
    {
        Layer* layer = layers[i];
        Option opt1 = opt;
        int cret = layer->create_pipeline(opt1);
        if (cret != 0)
        {
            printf("layer create_pipeline %d failed", (int)i);
            return -1;
        }
    }
    return 0;
}

int VulkanGraph::record_graph_pipeline()
{
    TEngine::VkCompute cmd(vkdev);

    if (!opt.blob_vkallocator)
    {
        local_blob_vkallocator = vkdev->acquire_blob_allocator();
        opt.blob_vkallocator = local_blob_vkallocator;
    }
    if (!opt.workspace_vkallocator)
    {
        opt.workspace_vkallocator = opt.blob_vkallocator;
    }
    if (!opt.staging_vkallocator)
    {
        local_staging_vkallocator = vkdev->acquire_staging_allocator();
        opt.staging_vkallocator = local_staging_vkallocator;
    }

    for (int i = 0; i < sgraph->graph->input_num; ++i)
    {
        const node_t input_node = get_graph_input_node(sgraph->graph, i);
        for (int k = 0; k < get_node_output_number(input_node); ++k)
        {
            const auto input_tensor = get_graph_input_tensor(sgraph->graph, i, k);
            const auto name = get_tensor_name(input_tensor);
            cmd.record_upload(tensor_map_[name], vktensor_map_[name], opt);
        }
    }

    Tensor input;
    Tensor output;

    for (size_t i = 0; i < layers.size(); i++)
    {
        Layer* layer = layers[i];
        std::string out_name = layer->tops[0];

        int cret = 0;
        if (layer->one_blob_only)
        {
            std::string const& in_name = layer->bottoms[0];
            auto& bottom_tensor = vktensor_map_[in_name];
            if (layer->support_inplace)
            {
                auto cret = layer->record_pipeline(bottom_tensor, cmd, opt);
                //FIXME: chec and log here
                vktensor_map_[out_name] = bottom_tensor;
            }
            else
            {
                VkTensor top_blob;
                cret = layer->record_pipeline(bottom_tensor, top_blob, cmd, opt);
                vktensor_map_[out_name] = top_blob;
            }
        }
        else
        {
            std::vector<VkTensor> bottom_blobs;
            for (auto const& inp : layer->bottoms)
            {
                bottom_blobs.push_back(vktensor_map_[inp]);
            }

            std::vector<VkTensor> top_blobs(1);
            cret = layer->record_pipeline(bottom_blobs, top_blobs, cmd, opt);
            vktensor_map_[out_name] = top_blobs.front();
        }

        if (cret != 0)
        {
            printf("layer record_pipeline %d failed", (int)i);
            return -1;
        }
    }

    auto output_layer = layers.back();
    auto const& name = output_layer->tops.front();
    cmd.record_download(vktensor_map_[name], output, opt);

    cmd.submit_and_wait();

    Tensor tmp_fp32;
    if (output.elemsize == output.elempack * 2)
    {
        TEngine::cast_float16_to_float32(output, tmp_fp32, opt);
    }
    else
    {
        tmp_fp32 = output;
    }

    Tensor blob_unpacked;
    if (opt.use_packing_layout)
    {
        convert_packing(tmp_fp32, blob_unpacked, 1, opt);
    }
    else
    {
        blob_unpacked = tmp_fp32;
    }

    tensor_map_[name]->data = blob_unpacked.data; // FIXME: leak?
    return 0;
}

int VulkanGraph::destory_pipeline()
{
    return 0;
}

} // namespace TEngine
