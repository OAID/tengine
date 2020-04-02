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
 * Author: jingyou@openailab.com
 */
#include "tm_serializer.hpp"
#include "tm1_serializer.hpp"

namespace TEngine {
namespace TMSerializer1 {

extern bool TmSerializerRegisterOpLoader1();
<<<<<<< HEAD

=======
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
}

using namespace TMSerializer1;

bool register_tm1_serializer(void)
{
    auto factory = TmSerializerFactory::GetFactory();

    factory->RegisterInterface<TmSerializer1>("tm_v1");
    auto tm_serializer = factory->Create("tm_v1");

    TmSerializerManager::SafeAdd("tm_v1", TmSerializerPtr(tm_serializer));

    return TmSerializerRegisterOpLoader1();
}

}    // namespace TEngine
<<<<<<< HEAD

=======
>>>>>>> bb35a6791dfd4a11405787254ac718ea8bb4d074
