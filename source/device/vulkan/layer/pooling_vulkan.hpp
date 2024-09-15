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

#ifndef LAYER_POOLING_HPP
#define LAYER_POOLING_HPP

#include "../vulkan_layer.hpp"
#include "../vulkan_command.hpp"
#include "padding_vulkan.hpp"

#include "pooling_param.h"

namespace TEngine {

class Pooling_vulkan : public Layer
{
public:
    Pooling_vulkan(ir_graph_t* ir_graph, ir_node_t* ir_node, const GPUDevice* vkdev);

    virtual int create_pipeline(const Option& opt);
    virtual int destroy_pipeline(const Option& opt);

    virtual int record_pipeline(const VkTensor& bottom_blob, VkTensor& top_blob, VkCompute& cmd, const Option& opt) const;

public:
    int pooling_type; // // 0:max    1:avg
    int kernel_h;     // = param_->kernel_h;
    int kernel_w;     // = param_->kernel_w;
    int stride_h;     // = param_->stride_h;
    int stride_w;     // = param_->stride_w;
    int global;       // = param_->global;
    int caffe_flavor; // = param_->caffe_flavor;
    int pad_h0;       // = param_->pad_h0;
    int pad_w0;       // = param_->pad_w0;
    int pad_h1;       // = param_->pad_h1;
    int pad_w1;       // = param_->pad_w1;
    int input_c;
    int input_h;
    int input_w;
    int output_c;
    int output_h;
    int output_w;

public:
    Padding_vulkan* padding;

    Pipeline* pipeline_pooling;
    Pipeline* pipeline_pooling_pack4;
    Pipeline* pipeline_pooling_pack8;
    Pipeline* pipeline_pooling_global;
    Pipeline* pipeline_pooling_global_pack4;
    Pipeline* pipeline_pooling_global_pack8;
};

} // namespace TEngine

#endif
