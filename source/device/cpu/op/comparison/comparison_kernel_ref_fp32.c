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
 * Author: ddzhao@openailab.com
 */

#include "comparison_kernel_ref.h"

void comp_equal(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
                p_comparison_param param, float* output)
{
    if (input1_count4 == 1)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (input0[i] == input1[0]);
        }
    }
    else if (input_count4 == input1_count4)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = ((*input0++) == (*input1++));
        }
    }
    else if (input_count4 == 1)
    {
        for (int i = 0; i < input1_count4; ++i)
        {
            *output++ = (input0[0] == input1[i]);
        }
    }
    else if (param->shape0[1] == input1_count4)
    {
        for (int n = 0; n < param->shape0[0]; n++)
        {
            for (int c = 0; c < param->shape0[1]; c++)
            {
                for (int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;

                    output[offset] = (input0[offset] == input1[c]);
                }
            }
        }
    }
    else if (param->shape1[1] == input_count4)
    {
        for (int n = 0; n < param->shape1[0]; n++)
        {
            for (int c = 0; c < param->shape1[1]; c++)
            {
                for (int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                    output[offset] = (input0[c] == input1[offset]);
                }
            }
        }
    }
}

void comp_nequal(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
                 p_comparison_param param, float* output)
{
    if (input1_count4 == 1)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (input0[i] != input1[0]);
        }
    }
    else if (input_count4 == input1_count4)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = ((*input0++) != (*input1++));
        }
    }
    else if (input_count4 == 1)
    {
        for (int i = 0; i < input1_count4; ++i)
        {
            *output++ = (input0[0] != input1[i]);
        }
    }
    else if (param->shape0[1] == input1_count4)
    {
        for (int n = 0; n < param->shape0[0]; n++)
        {
            for (int c = 0; c < param->shape0[1]; c++)
            {
                for (int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;

                    output[offset] = (input0[offset] != input1[c]);
                }
            }
        }
    }
    else if (param->shape1[1] == input_count4)
    {
        for (int n = 0; n < param->shape1[0]; n++)
        {
            for (int c = 0; c < param->shape1[1]; c++)
            {
                for (int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                    output[offset] = (input0[c] != input1[offset]);
                }
            }
        }
    }
}

void comp_less(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
               p_comparison_param param, float* output)
{
    if (input1_count4 == 1)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (input0[i] < input1[0]);
        }
    }
    else if (input_count4 == input1_count4)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = ((*input0++) < (*input1++));
        }
    }
    else if (input_count4 == 1)
    {
        for (int i = 0; i < input1_count4; ++i)
        {
            *output++ = (input0[0] < input1[i]);
        }
    }
    else if (param->shape0[1] == input1_count4)
    {
        for (int n = 0; n < param->shape0[0]; n++)
        {
            for (int c = 0; c < param->shape0[1]; c++)
            {
                for (int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;

                    output[offset] = (input0[offset] < input1[c]);
                }
            }
        }
    }
    else if (param->shape1[1] == input_count4)
    {
        for (int n = 0; n < param->shape1[0]; n++)
        {
            for (int c = 0; c < param->shape1[1]; c++)
            {
                for (int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                    output[offset] = (input0[c] < input1[offset]);
                }
            }
        }
    }
}

void comp_lesse(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
                p_comparison_param param, float* output)
{
    if (input1_count4 == 1)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (input0[i] <= input1[0]);
        }
    }
    else if (input_count4 == input1_count4)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = ((*input0++) <= (*input1++));
        }
    }
    else if (input_count4 == 1)
    {
        for (int i = 0; i < input1_count4; ++i)
        {
            *output++ = (input0[0] <= input1[i]);
        }
    }
    else if (param->shape0[1] == input1_count4)
    {
        for (int n = 0; n < param->shape0[0]; n++)
        {
            for (int c = 0; c < param->shape0[1]; c++)
            {
                for (int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;

                    output[offset] = (input0[offset] <= input1[c]);
                }
            }
        }
    }
    else if (param->shape1[1] == input_count4)
    {
        for (int n = 0; n < param->shape1[0]; n++)
        {
            for (int c = 0; c < param->shape1[1]; c++)
            {
                for (int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                    output[offset] = (input0[c] <= input1[offset]);
                }
            }
        }
    }
}

void comp_greater(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
                  p_comparison_param param, float* output)
{
    if (input1_count4 == 1)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (input0[i] > input1[0]);
        }
    }
    else if (input_count4 == input1_count4)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = ((*input0++) > (*input1++));
        }
    }
    else if (input_count4 == 1)
    {
        for (int i = 0; i < input1_count4; ++i)
        {
            *output++ = (input0[0] > input1[i]);
        }
    }
    else if (param->shape0[1] == input1_count4)
    {
        for (int n = 0; n < param->shape0[0]; n++)
        {
            for (int c = 0; c < param->shape0[1]; c++)
            {
                for (int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;

                    output[offset] = (input0[offset] > input1[c]);
                }
            }
        }
    }
    else if (param->shape1[1] == input_count4)
    {
        for (int n = 0; n < param->shape1[0]; n++)
        {
            for (int c = 0; c < param->shape1[1]; c++)
            {
                for (int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                    output[offset] = (input0[c] > input1[offset]);
                }
            }
        }
    }
}

void comp_greatere(int input_hw, int input_hw_1, int input_count4, int input1_count4, float* input0, float* input1,
                   p_comparison_param param, float* output)
{
    if (input1_count4 == 1)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = (input0[i] >= input1[0]);
        }
    }
    else if (input_count4 == input1_count4)
    {
        for (int i = 0; i < input_count4; ++i)
        {
            *output++ = ((*input0++) >= (*input1++));
        }
    }
    else if (input_count4 == 1)
    {
        for (int i = 0; i < input1_count4; ++i)
        {
            *output++ = (input0[0] >= input1[i]);
        }
    }
    else if (param->shape0[1] == input1_count4)
    {
        for (int n = 0; n < param->shape0[0]; n++)
        {
            for (int c = 0; c < param->shape0[1]; c++)
            {
                for (int i = 0; i < input_hw; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape0[1] * input_hw + c * input_hw + i;
                    else
                        offset = n * param->shape0[1] * input_hw + i * param->shape0[1] + c;

                    output[offset] = (input0[offset] >= input1[c]);
                }
            }
        }
    }
    else if (param->shape1[1] == input_count4)
    {
        for (int n = 0; n < param->shape1[0]; n++)
        {
            for (int c = 0; c < param->shape1[1]; c++)
            {
                for (int i = 0; i < input_hw_1; ++i)
                {
                    int offset = 0;
                    if (param->layout == 0)
                        offset = n * param->shape1[1] * input_hw_1 + c * input_hw_1 + i;
                    else
                        offset = n * param->shape1[1] * input_hw_1 + i * param->shape1[1] + c;

                    output[offset] = (input0[c] >= input1[offset]);
                }
            }
        }
    }
}

int ref_comparison_fp32(float* input0, float* input1, float* output, p_comparison_param param)
{
    int input_hw = param->shape0[2] * param->shape0[3];
    int input_hw_1 = param->shape1[2] * param->shape1[3];
    int input_count4 = param->shape0[0] * param->shape0[1] * input_hw;
    int input1_count4 = param->shape1[0] * param->shape1[1] * input_hw_1;

    switch (param->type)
    {
    case 0:
    {
        comp_equal(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
        break;
    }
    case 1:
    {
        comp_nequal(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
        break;
    }
    case 2:
    {
        comp_greater(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
        break;
    }
    case 3:
    {
        comp_greatere(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
        break;
    }
    case 4:
    {
        comp_less(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
        break;
    }
    case 5:
    {
        comp_lesse(input_hw, input_hw_1, input_count4, input1_count4, input0, input1, param, output);
        break;
    }
    default:
        return -1;
        break;
    }
    return 0;
}
