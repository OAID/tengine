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
 * Copyright (c) 2017, Open AI Lab
 * Author: haitao@openailab.com
 */
#ifndef __GRAPH_OPTIMIZER_HPP__
#define __GRAPH_OPTIMIZER_HPP__

#include <functional>
#include <string>

#include "any.hpp"
#include "simple_object_manager.hpp"

namespace TEngine {

class Graph;
struct GraphOptimizer;

using graph_opt_t = std::function<bool(Graph *, GraphOptimizer *)>;

struct GraphOptimizer {
  std::string name;
  graph_opt_t optimizer;
  any args;
};

class GraphOptimizerManager
    : public SimpleObjectManager<GraphOptimizerManager, GraphOptimizer *> {
 public:
  static bool RunOpt(const std::string &name, Graph *graph);

  static void Init(void);
};

}  // namespace TEngine

#endif
