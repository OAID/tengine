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

#include "vulkan_layer.hpp"

namespace TEngine {

Layer::Layer(const GPUDevice* vkdev)
    : vkdev(vkdev), one_blob_only(true), support_inplace(false)
{
}

Layer::~Layer()
{
}

int Layer::create_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Layer::destroy_pipeline(const Option& /*opt*/)
{
    return 0;
}

int Layer::upload_model(VkTransfer& cmd, const Option& opt)
{
    return 0;
}

int Layer::record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const
{
    return 0;
}

int Layer::record_pipeline(VkTensor& bottom_top_blob, VkCompute& cmd, const Option& opt) const
{
    return 0;
}

int Layer::record_pipeline(const std::vector<VkTensor>& bottom_blobs, std::vector<VkTensor>& top_blobs, VkCompute& cmd, const Option& opt) const
{
    printf("run layer record_pipeline VkTensors\n");
    return 0;
}

} // namespace TEngine
