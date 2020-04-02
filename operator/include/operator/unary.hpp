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
 * Copyright (c) 2019, Open AI Lab
 * Author: bzhang@openailab.com
 */
#ifndef __UNARY_HPP__
#define __UNARY_HPP__

#include "operator.hpp"
#include "unary_param.hpp"
namespace TEngine {

class Unary : public OperatorWithParam<Unary, UnaryParam>
{
public:
    Unary()
    {
        name_ = "Unary";
    }
    Unary(const Unary& src) = default;
    virtual ~Unary(){};

    void SetSchema(void) override;

    //bool InferShape(const std::vector<TShape>& ishape, std::vector<TShape>& oshape, int layout) override;
};

}    // namespace TEngine

#endif
