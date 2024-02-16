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
 * Revised: lswang@openailab.com
 */

#pragma once

#include "cpu_define.h"

#include <stdint.h>
#include <stdbool.h>

struct node;
struct node_ops;
struct exec_node;
struct exec_graph;

struct exec_node
{
    struct node* ir_node;
    struct node_ops* node_ops;
    void* ops_priv; /* priv data for ops */

    int8_t inplace_map_num;
    int8_t output_num;

    union
    {
        uint8_t* inplace_map_ptr;
        uint8_t inplace_map[4]; /* opt for single inplace map, such as relu */
    };

    union
    {
        int8_t block_id[4];
        int8_t* block_id_ptr;
    };

    int shared_mem_size;
    int shared_pack4_mem_size;
};

struct node_ops
{
    int (*prerun)(struct node_ops*, struct exec_node*, struct exec_graph*);
    int (*run)(struct node_ops*, struct exec_node*, struct exec_graph*);
    int (*reshape)(struct node_ops*, struct exec_node*, struct exec_graph*);
    int (*postrun)(struct node_ops*, struct exec_node*, struct exec_graph*);

    /* for private data of each node */
    /* init node is called when a node_ops is bound to an exec_node,
       just before prerun() is called.
       so that the node_ops can set the shared_mem needed in this function,
       as well as set the ops_priv in exec_node.
    */
    int (*init_node)(struct node_ops*, struct exec_node*, struct exec_graph*);

    /* release node is called after postrun() is called */
    int (*release_node)(struct node_ops*, struct exec_node*, struct exec_graph*);

    /* score */
    int (*score)(struct node_ops*, struct exec_graph*, struct node*);

    /* is ref op */
    bool is_ref_op;
};

int init_exec_node(struct exec_graph* exec_graph, struct exec_node* exec_node, struct node* ir_node, struct node_ops* node_ops);
void release_exec_node(struct exec_graph* exec_graph, struct exec_node* exec_node, struct node_ops* node_ops);
