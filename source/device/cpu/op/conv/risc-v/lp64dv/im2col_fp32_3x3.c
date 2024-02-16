#include "vsetvl_rvv.h"

void im2col_fp32_3x3(const float* input, const int input_x, const int input_y, const int input_channels, float* col, const int stride)
{
    vsetvl_e32_m2();
    const int in_xy = input_x * input_y;
    const float* row0 = input;
    const float* row1 = row0 + input_x;
    const float* row2 = row1 + input_x;
    float* cur_col = col;

    if (stride == 1)
    {
        for (int c = 0; c < input_channels; ++c)
        {
            asm("vle32.v    v0, (%0);\n"
                "vle32.v    v2, (%1);\n"
                "vle32.v    v4, (%2);\n"

                "addi       t0,  %0, 4;\n"
                "addi       t1,  %0, 8;\n"

                "vle32.v    v6, (t0);\n"
                "vle32.v    v8, (t1);\n"

                "addi       t0,  %1, 4;\n"
                "addi       t1,  %1, 8;\n"

                "vle32.v    v10, (t0);\n"
                "vle32.v    v12, (t1);\n"

                "addi       t0, %2, 4;\n"
                "addi       t1, %2, 8;\n"

                "vle32.v    v14, (t0);\n"
                "vle32.v    v16, (t1);\n"

                "vse32.v    v0, (%3);\n"
                "addi       t0, %3, 32;\n"
                "vse32.v    v6, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v8, (t0);\n"
                "addi       t0, t0, 32;\n"

                "vse32.v    v2, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v10, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v12, (t0);\n"
                "addi       t0, t0, 32;\n"

                "vse32.v    v4, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v14, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v16, (t0);\n"
                "addi       t0, t0, 32;\n"
                :
                : "r"(row0), "r"(row1), "r"(row2), "r"(cur_col)
                : "t0", "t1", "memory");

            row0 += in_xy;
            row1 += in_xy;
            row2 += in_xy;
            cur_col += 72;
        }
    }
    else
    {
        for (int c = 0; c < input_channels; ++c)
        {
            asm("li         t0, 8;\n"
                "vlse32.v   v0, (%0), t0;\n"
                "add        t1, %0, 0x4;\n"
                "vlse32.v   v2, (t1), t0;\n"
                "add        t1, t1, 0x4;\n"
                "vlse32.v   v4, (t1), t0;\n"

                "vlse32.v   v6, (%1), t0;\n"
                "add        t1, %1, 0x4;\n"
                "vlse32.v   v8, (t1), t0;\n"
                "add        t1, t1, 0x4;\n"
                "vlse32.v   v10, (t1), t0;\n"

                "vlse32.v   v12, (%2), t0;\n"
                "add        t1, %2, 0x4;\n"
                "vlse32.v   v14, (t1), t0;\n"
                "add        t1, t1, 0x4;\n"
                "vlse32.v   v16, (t1), t0;\n"

                "vse32.v    v0, (%3);\n"
                "addi       t0, %3, 32;\n"
                "vse32.v    v2, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v4, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v6, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v8, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v10, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v12, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v14, (t0);\n"
                "addi       t0, t0, 32;\n"
                "vse32.v    v16, (t0);\n"
                :
                : "r"(row0), "r"(row1), "r"(row2), "r"(cur_col)
                : "t0", "t1", "memory");
            row0 += in_xy;
            row1 += in_xy;
            row2 += in_xy;
            cur_col += 72;
        }
    }
}
