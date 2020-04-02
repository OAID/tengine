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
 * Author: zpluo@openailab.com
 */
#ifndef __REDUCTION_PARAM_HPP__
#define __REDUCTION_PARAM_HPP__

#include "parameter.hpp"

namespace TEngine {

struct ReductionParam : public NamedParam
{
    int dim_0;
    int dim_1;
    int dim_2;
    int dim_3;
<<<<<<< HEAD
    //type : 0: sum; 1: mean.
=======
    // type : 0: sum; 1: mean.
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
    int type;
    int keepdim;
    DECLARE_PARSER_STRUCTURE(ReductionParam)
    {
        DECLARE_PARSER_ENTRY(dim_0);
        DECLARE_PARSER_ENTRY(dim_1);
        DECLARE_PARSER_ENTRY(dim_2);
        DECLARE_PARSER_ENTRY(dim_3);
        DECLARE_PARSER_ENTRY(keepdim);
        DECLARE_PARSER_ENTRY(type);
    };
};

}    // namespace TEngine

#endif