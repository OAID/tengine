#include "vsetvl_rvv.h"

// FIXME: optimize vectorize loop
void im2col_fp32_1x1_tile8(const float* input, const int input_xy, const int input_channels, float* col)
{
    vsetvl_e32_m2();

    const float* c0 = input;
    const float* c1 = input + input_xy;
    const int input_xy_stride = 2 * input_xy;

    float* o0 = col;
    float* o1 = col + 8;

    int c = 0;
    for (; c < (input_channels & -2); c += 2)
    {
        __asm__(
            "vle32.v    v0, (%0); \n"
            "vle32.v    v2, (%1); \n"
            "vse32.v    v0, (%2); \n"
            "vse32.v    v2, (%3); \n"
            :
            : "r"(c0), "r"(c1), "r"(o0), "r"(o1)
            : "memory");
        o0 += 16;
        o1 += 16;
        c0 += input_xy_stride;
        c1 += input_xy_stride;
    }

    if (c < input_channels)
    {
        __asm__("vle32.v    v0, (%0);\n"
                "vse32.v    v0, (%1);\n"
                :
                : "r"(c0), "r"(o0)
                : "memory");
    }
}
