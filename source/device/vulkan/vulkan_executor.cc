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

#include "vulkan_executor.hpp"
#include "vulkan_helper.hpp"
#include "vulkan_gpu.hpp"
#include "vulkan_graph.hpp"

extern "C"
{
#include "operator/op.h"
#include "convolution_param.h"
}

using namespace TEngine;

bool VULKANEngine::init()
{
    return true;
}



int VULKANEngine::VULKANEnginePreRun(struct subgraph* subgraph)
{
    // TLOG_INFO("==== vulkan prerun start ====\n");
    // struct device *vk_dev = (struct device *)dev;
    struct graph *orig_graph = subgraph->graph;
    // struct vk_dev_priv *priv = (struct vk_dev_priv *)orig_graph->dev_priv;

    // /* todo: set the tensor shape ? */

    // /* create exec_graph */
    VulkanGraph* vk_exec_graph = new VulkanGraph(subgraph);

    if (vk_exec_graph == nullptr)
    {
        // set_tengine_errno(EIO);
        TLOG_ERR("vulkan exec graph is NULL\n");
        return -1;        
    }

    vk_exec_graph->upload_model();
    vk_exec_graph->create_pipeline();

    subgraph->device_graph = vk_exec_graph;

    int node_num = subgraph->node_num;
    TLOG_INFO("==== vulkan prerun done ====\n");
    return 0;

};

int VULKANEngine::VULKANEngineRun(struct subgraph* subgraph)
{
    // struct vk_device *vk_dev = (struct vk_device *)dev; 
    struct graph *orig_graph = subgraph->graph;
    // struct vk_dev_priv *priv = (struct vk_dev_priv *)orig_graph->dev_priv;

    VulkanGraph *vk_exec_graph = (VulkanGraph *)subgraph->device_graph;
    if (vk_exec_graph == nullptr)
    {
        // set_tengine_errno(EIO);
        TLOG_ERR("vulkan exec graph is NULL\n");
        return -1;
    }

    vk_exec_graph->record_graph_pipeline();
    return 0;
}

void VULKANEngine::VULKANEnginePostRun()
{
    return;
};
