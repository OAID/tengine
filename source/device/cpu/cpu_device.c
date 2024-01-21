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
 * Author: haitao@openailab.com
 */

#include "cpu_device.h"

#include "defines.h"

#include "cpu_node.h"
#include "cpu_graph.h"
#include "cpu_pool.h"
#include "cpu_dump.h"

#include "device/cpu/cpu_ops.h"

#include "graph/tensor.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/subgraph.h"
#include "optimizer/split.h"
#include "module/module.h"
#include "serializer/serializer.h"
#include "utility/vector.h"
#include "utility/sys_port.h"
#include "utility/utils.h"
#include "utility/log.h"

#include <string.h>

int init_cpu(struct device* device)
{
    (void)device;
    return register_all_cpu_ops();
}

int release_cpu(struct device* device)
{
    (void)device;
    return unregister_all_cpu_ops();
}

static int prerun(struct device* dev, struct subgraph* subgraph, void* option)
{
    struct exec_graph* exec_graph;
    struct cpu_option* opt = (struct cpu_option*)option;

    /* create exec_graph */
    exec_graph = create_exec_graph(subgraph, opt->num_thread, opt->precision, opt->affinity);

    if (exec_graph == NULL)
        return -1;

    if (alloc_exec_graph_mem(exec_graph) < 0 || prerun_exec_graph(exec_graph) < 0)
    {
        release_exec_graph(exec_graph);
        return -1;
    }

    const char* env = getenv(TENGINE_PRINT_LAYER_COST);
    if (env && env[0] == '1')
    {
        int node_num = get_vector_num(exec_graph->exec_node_list);
        double* time = (double*)sys_malloc(sizeof(double) * (node_num + 2)); // 0~num-1 for node, num for repeat, num + 1 for sum
        memset(time, 0, sizeof(double) * (node_num + 2));
        exec_graph->timer = time;
    }
    else
    {
        exec_graph->timer = NULL;
    }

    subgraph->device_graph = exec_graph;

    return 0;
}

static int run(struct device* dev, struct subgraph* subgraph)
{
    struct exec_graph* exec_graph = (struct exec_graph*)subgraph->device_graph;

    int node_num = get_vector_num(exec_graph->exec_node_list);

    if (exec_graph->timer)
    {
        double* timer = (double*)exec_graph->timer;
        timer[node_num] += 1.0; // repeat
    }

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* node = (struct exec_node*)get_vector_data(exec_graph->exec_node_list, i);
        struct node_ops* node_ops = node->node_ops;

        /* TODO: handle the shape changed  and dynamic shape case */
        if (node_ops->reshape && node_ops->reshape(node_ops, node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to reshape node %d, %s\n", dev->name, node->ir_node->index, node->ir_node->name);
            return -1;
        }

        /* TODO: add dynamic skip feature */
#ifdef DEBUG_TIME
        double start = get_current_time();
#endif
        double st_time, end_time;
        if (exec_graph->timer)
        {
            st_time = get_current_time();
        }
        if (node_ops->run(node_ops, node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to run node %d, %s\n", dev->name, node->ir_node->index, node->ir_node->name);
            return -1;
        }
        char* name = node->ir_node->name;
#ifdef DEBUG_TIME
        double end = get_current_time();
        fprintf(stderr, "%-20s  %8.2f ms  %s\n", get_op_name_from_type(node->ir_node->op.type), end - start, name);
#endif
        if (exec_graph->timer)
        {
            end_time = get_current_time();
            double* timer = (double*)exec_graph->timer;
            double cur_time = end_time - st_time;

            // save min time
            if (timer[node_num] < 2.0)
            {
                timer[i] = cur_time;
            }
            else
            {
                timer[i] = cur_time < timer[i] ? cur_time : timer[i];
            }
            timer[node_num + 1] += cur_time; // sum
        }
#ifdef DEBUG_DATA
        struct graph* ir_graph = node->ir_node->graph;

        for (uint8_t j = 0; j < node->ir_node->input_num; j++)
        {
            struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->ir_node->input_tensors[j]);
            if (input_tensor->dim_num <= 5)
            {
                char dir_str[32] = {0};
                sprintf(dir_str, "in[%d]", j);

                if (NULL != input_tensor->data)
                {
                    extract_feature_from_tensor(dir_str, name, input_tensor);
                }
            }
        }

        for (uint8_t j = 0; j < node->ir_node->output_num; j++)
        {
            struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->ir_node->output_tensors[j]);
            /* debug */
            if (output_tensor->dim_num <= 5)
            {
                char dir_str[32] = {0};
                sprintf(dir_str, "out[%d]", j);

                extract_feature_from_tensor(dir_str, name, output_tensor);
            }
        }
#endif
        const char* env = getenv(TENGINE_DUMP_LAYER);
        if (env && env[0] == '1')
        {
            struct graph* ir_graph = node->ir_node->graph;
            struct tensor* input_tensor = get_ir_graph_tensor(ir_graph, node->ir_node->input_tensors[0]);
            struct tensor* output_tensor = get_ir_graph_tensor(ir_graph, node->ir_node->output_tensors[0]);
            /* debug */
            if (input_tensor->dim_num <= 5)
                extract_feature_from_tensor("in", name, input_tensor);
            if (output_tensor->dim_num <= 5)
                extract_feature_from_tensor("out", name, output_tensor);
        }

//#define DUMP_NODE_OUTPUT
#ifdef DUMP_NODE_OUTPUT
        /* dump the node output */
        struct node* ir_node = node->ir_node;
        struct graph* ir_graph = ir_node->graph;

        for (int i = 0; i < ir_node->input_num; i++)
        {
            char fname[128];
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->input_tensors[i]);

            sprintf(fname, "/tmp/dump/node%s%d.%d", (ir_node->idx < 10 ? "0" : ""), ir_node->idx, i);

            dump_float(fname, ir_tensor->data, ir_tensor->elem_num);
        }

#endif
#if 0
        struct node* ir_node = node->ir_node;
        struct graph* ir_graph = ir_node->graph;
        for (int i = 0; i < ir_node->output_num; ++i)
        {
            struct tensor* ir_tensor = get_ir_graph_tensor(ir_graph, ir_node->output_tensors[i]);
            float mean = tensor_mean(ir_tensor);
            fprintf(stderr, "%s output %d, mean: %f\n", ir_node->name, i, mean);
        }
#endif
    }

    return 0;
}

static int postrun(struct device* dev, struct subgraph* subgraph)
{
    struct exec_graph* exec_graph = (struct exec_graph*)subgraph->device_graph;

    int node_num = get_vector_num(exec_graph->exec_node_list);

    for (int i = 0; i < node_num; i++)
    {
        struct exec_node* node = (struct exec_node*)get_vector_data(exec_graph->exec_node_list, i);
        struct node_ops* node_ops = node->node_ops;

        if (exec_graph->timer)
        {
            extract_node_executed_time(subgraph, i);
        }

        if (node_ops->postrun && node_ops->postrun(node_ops, node, exec_graph) < 0)
        {
            TLOG_ERR("%s: failed to postrun node %d\n", dev->name, node->ir_node->index);
        }
    }

    release_exec_graph(exec_graph);

    subgraph->device_graph = NULL;

    return 0;
}

static int cpu_dev_release_exec_graph(struct device* dev, void* exec_graph)
{
    if (NULL != exec_graph)
    {
        release_exec_graph(exec_graph);
    }

    return 0;
}

static int cpu_allocate(struct device* device, struct subgraph* sub_graph)
{
    /* set the correct input wait count: INPUT tensor is always ready */
    sub_graph->input_wait_count = 0;

    for (int i = 0; i < sub_graph->input_num; i++)
    {
        struct tensor* tensor = get_ir_graph_tensor(sub_graph->graph, sub_graph->input_tensor_list[i]);

        if (tensor->tensor_type == TENSOR_TYPE_VAR)
            sub_graph->input_wait_count++;
    }

    return 0;
}

static int cpu_describe(struct device* device, struct vector* allowed_ops, struct vector* blocked_ops, struct vector* precision)
{
    if (NULL == device)
    {
        return -1;
    }

    if (NULL == allowed_ops)
    {
        TLOG_ERR("Error: Allowed op list pointer is NULL\n");
    }
    if (NULL == blocked_ops)
    {
        TLOG_ERR("Error: Blocked op list pointer is NULL\n");
    }

    for (int i = OP_GENERIC + 1; i < OP_BUILTIN_LAST - 1; i++)
    {
        push_vector_data(allowed_ops, &i);
    }

    int precision_var = TENGINE_DT_FP32;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_FP16;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_UINT8;
    push_vector_data(precision, &precision_var);
    precision_var = TENGINE_DT_INT8;
    push_vector_data(precision, &precision_var);

    return 0;
}

static int cpu_evaluation(struct device* device, struct subgraph* sub_graph, struct vector* tensor, struct vector* node)
{
    if (NULL == device)
    {
        return -1;
    }

    (void)sub_graph;
    (void)tensor;
    (void)node;

    return 0;
}

static int cpu_release(struct device* device, struct subgraph* sub_graph)
{
    if (NULL == device)
    {
        return -1;
    }

    (void)sub_graph;

    return 0;
}

int cpu_split_graph(struct graph* ir_graph)
{
    struct device* default_device = find_default_device();

    struct subgraph* cpu_graph = (struct subgraph*)sys_malloc(sizeof(struct subgraph));

    init_ir_subgraph(ir_graph, cpu_graph, 0);

    cpu_graph->input_num = (uint8_t)ir_graph->input_num;
    cpu_graph->output_num = (uint8_t)ir_graph->output_num;

    cpu_graph->node_num = ir_graph->node_num;
    cpu_graph->node_list = (uint16_t*)sys_malloc(sizeof(uint16_t) * cpu_graph->node_num);

    for (uint16_t i = 0; i < cpu_graph->node_num; i++)
    {
        cpu_graph->node_list[i] = ir_graph->node_list[i]->index;
    }

    cpu_graph->device = default_device;

    push_vector_data(ir_graph->subgraph_list, &cpu_graph);

    //
    generate_sub_graph_io(ir_graph);
    add_sub_graph_to_ir_graph(ir_graph);

    // add node sub graph id
    for (int i = 0; i < (uint16_t)get_vector_num(ir_graph->subgraph_list); i++)
    {
        struct subgraph* sub_graph = *(struct subgraph**)get_vector_data(ir_graph->subgraph_list, i);
        sub_graph->index = i;

        for (uint16_t j = 0; j < sub_graph->node_num; j++)
        {
            uint16_t node_id = sub_graph->node_list[j];
            struct node* ir_node = get_ir_graph_node(ir_graph, node_id);
            ir_node->subgraph_idx = sub_graph->index;
        }
    }

    return 0;
}

static struct interface cpu_interface = {
    .init = init_cpu,
    .pre_run = prerun,
    .run = run,
    .post_run = postrun,
    .async_run = NULL,
    .async_wait = NULL,
    .release_graph = cpu_dev_release_exec_graph,
    .release_device = release_cpu,
};

static struct allocator cpu_allocator = {
    .describe = cpu_describe,
    .evaluation = cpu_evaluation,
    .allocate = cpu_allocate,
    .release = cpu_release,
};

static struct optimizer cpu_optimizer = {
    .split_graph = cpu_split_graph,
    .optimize_graph = NULL,
};

static struct cpu_device cpu_dev = {
    .base = {
        .name = CPU_DEVICE_NAME,
        .interface = &cpu_interface,
        .allocator = &cpu_allocator,
        .optimizer = &cpu_optimizer,
        .scheduler = NULL,
        .privacy = NULL,
    },
    .master_cpu = 0,
    .cpu_model = 0,
};

int register_cpu_device(void)
{
#ifdef TENGINE_AUTO_LOAD_HCL
    dlopen("libtengine_hcl.so", RTLD_NOW);
#endif

    int ret = register_device(&cpu_dev.base);
    if (0 != ret)
    {
        TLOG_INFO("Tengine plugin %s register failed.\n", cpu_dev.base.name);
        return ret;
    }

    TLOG_INFO("Tengine plugin device %s is registered.\n", cpu_dev.base.name);
    return 0;
}

int unregister_cpu_device(void)
{
    int ret = unregister_device(&cpu_dev.base);
    if (0 != ret)
    {
        TLOG_INFO("Tengine plugin %s unregister failed.\n", cpu_dev.base.name);
        return ret;
    }

    TLOG_INFO("Tengine plugin device %s is unregistered.\n", cpu_dev.base.name);
    return 0;
}
