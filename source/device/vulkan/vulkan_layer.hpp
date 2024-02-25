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

#ifndef VULKAN_LAYER_HPP
#define VULKAN_LAYER_HPP

#include <vulkan/vulkan.h>
#include "vulkan_command.hpp"
#include "vulkan_pipeline.hpp"

extern "C" {
#include "api/c_api.h"
#include "device/device.h"
#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "executer/executer.h"
#include "optimizer/split.h"
#include "module/module.h"
#include "utility/vector.h"
#include "utility/log.h"
}

namespace TEngine {

class Layer
{
public:
    // empty
    Layer(const GPUDevice* vkdev);
    // virtual destructor
    virtual ~Layer();

    // layer implementation specific setup
    // return 0 if success
    virtual int create_pipeline(const Option& opt);

    // layer implementation specific clean
    // return 0 if success
    virtual int destroy_pipeline(const Option& opt);

    // upload weight blob from host to device
    virtual int upload_model(VkTransfer& cmd, const Option& opt);

    // virtual int record_pipeline(VkCompute& cmd, const Option& opt) const;
    virtual int record_pipeline(VkTensor& bottom_top_blob, VkCompute& cmd, const Option& opt) const;
    virtual int record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const;

    virtual int record_pipeline(const std::vector<VkTensor>& bottom_blobs, std::vector<VkTensor>& top_blobs, VkCompute& cmd, const Option& opt) const;

public:
    // accept input blob with packed storage
    bool support_packing;

    // accept bf16
    bool support_bf16_storage;

    bool one_blob_only;
    bool support_inplace;

public:
    const GPUDevice* vkdev;
    std::vector<std::string> bottoms;
    std::vector<std::string> tops;

public:
    // Node* node;
    ir_graph_t* graph;
    ir_node_t* node;
};

Layer* create_layer(std::string type);

} // namespace TEngine

#endif // VULKAN_LAYER_HPP
