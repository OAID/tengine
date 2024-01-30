#include "vsetvl_rvv.h"

void sgemm_8x8_rv64(const float* cur_col, const float* cur_kernel, const float* bias, const int act, float* cur_output, const int output_xy, const int kernel_size, const int n)
{
    vsetvl_e32_m2();

    // v16 ~ v30: result of c0 ~ v7
    if (bias)
    {
        asm("vle32.v        v0, (%0);\n"
            "vrgather.vi    v16, v0, 0;\n"
            "vrgather.vi    v18, v0, 1;\n"
            "vrgather.vi    v20, v0, 2;\n"
            "vrgather.vi    v22, v0, 3;\n"
            "vrgather.vi    v24, v0, 4;\n"
            "vrgather.vi    v26, v0, 5;\n"
            "vrgather.vi    v28, v0, 6;\n"
            "vrgather.vi    v30, v0, 7;\n"
            :
            : "r"(bias));
    }
    else
    {
        asm(
            "vmv.v.x      v16,    x0;\n"
            "vmv.v.x      v18,    x0;\n"
            "vmv.v.x      v20,    x0;\n"
            "vmv.v.x      v22,    x0;\n"
            "vmv.v.x      v24,    x0;\n"
            "vmv.v.x      v26,    x0;\n"
            "vmv.v.x      v28,    x0;\n"
            "vmv.v.x      v30,    x0;\n");
    }

    const float* k0 = cur_kernel;
    const float* k1 = k0 + 8;
    const float* k2 = k1 + 8;
    const float* k3 = k2 + 8;

    const float* col0 = cur_col;
    const float* col1 = col0 + 8;
    const float* col2 = col1 + 8;
    const float* col3 = col2 + 8;

    int k = 0;
    for (; k < (kernel_size & -4); k += 4)
    {
        asm(
            "vle32.v      v0,   (%0);\n"
            "vle32.v      v2,   (%4);\n"
            "vle32.v      v4,   (%1);\n"
            "vle32.v      v6,   (%5);\n"

            "vrgather.vi  v8,    v2, 0;\n"
            "vrgather.vi  v10,   v2, 1;\n"
            "vrgather.vi  v12,   v2, 2;\n"
            "vrgather.vi  v14,   v2, 3;\n"

            "vfmacc.vv    v16,   v0, v8;\n"
            "vfmacc.vv    v18,   v0, v10;\n"
            "vfmacc.vv    v20,   v0, v12;\n"
            "vfmacc.vv    v22,   v0, v14;\n"

            "vrgather.vi  v8,    v2, 4;\n"
            "vrgather.vi  v10,   v2, 5;\n"
            "vrgather.vi  v12,   v2, 6;\n"
            "vrgather.vi  v14,   v2, 7;\n"

            "vfmacc.vv    v24,   v0, v8;\n"
            "vfmacc.vv    v26,   v0, v10;\n"
            "vfmacc.vv    v28,   v0, v12;\n"
            "vfmacc.vv    v30,   v0, v14;\n"

            "vrgather.vi  v8,    v6, 0;\n"
            "vrgather.vi  v10,   v6, 1;\n"
            "vrgather.vi  v12,   v6, 2;\n"
            "vrgather.vi  v14,   v6, 3;\n"

            "vfmacc.vv    v16,   v4, v8;\n"
            "vfmacc.vv    v18,   v4, v10;\n"
            "vfmacc.vv    v20,   v4, v12;\n"
            "vfmacc.vv    v22,   v4, v14;\n"

            "vrgather.vi  v8,    v6, 4;\n"
            "vrgather.vi  v10,   v6, 5;\n"
            "vrgather.vi  v12,   v6, 6;\n"
            "vrgather.vi  v14,   v6, 7;\n"

            "vfmacc.vv    v24,   v4, v8;\n"
            "vfmacc.vv    v26,   v4, v10;\n"
            "vfmacc.vv    v28,   v4, v12;\n"
            "vfmacc.vv    v30,   v4, v14;\n"

            "vle32.v      v0,     (%2); \n"
            "vle32.v      v2,     (%6); \n"
            "vle32.v      v4,     (%3); \n"
            "vle32.v      v6,     (%7); \n"

            "vrgather.vi  v8,    v2, 0;\n"
            "vrgather.vi  v10,   v2, 1;\n"
            "vrgather.vi  v12,   v2, 2;\n"
            "vrgather.vi  v14,   v2, 3;\n"

            "vfmacc.vv    v16,   v0, v8;\n"
            "vfmacc.vv    v18,   v0, v10;\n"
            "vfmacc.vv    v20,   v0, v12;\n"
            "vfmacc.vv    v22,   v0, v14;\n"

            "vrgather.vi  v8,    v2, 4;\n"
            "vrgather.vi  v10,   v2, 5;\n"
            "vrgather.vi  v12,   v2, 6;\n"
            "vrgather.vi  v14,   v2, 7;\n"

            "vfmacc.vv    v24,   v0, v8;\n"
            "vfmacc.vv    v26,   v0, v10;\n"
            "vfmacc.vv    v28,   v0, v12;\n"
            "vfmacc.vv    v30,   v0, v14;\n"

            "vrgather.vi  v8,    v6, 0;\n"
            "vrgather.vi  v10,   v6, 1;\n"
            "vrgather.vi  v12,   v6, 2;\n"
            "vrgather.vi  v14,   v6, 3;\n"

            "vfmacc.vv    v16,   v4, v8;\n"
            "vfmacc.vv    v18,   v4, v10;\n"
            "vfmacc.vv    v20,   v4, v12;\n"
            "vfmacc.vv    v22,   v4, v14;\n"

            "vrgather.vi  v8,    v6, 4;\n"
            "vrgather.vi  v10,   v6, 5;\n"
            "vrgather.vi  v12,   v6, 6;\n"
            "vrgather.vi  v14,   v6, 7;\n"

            "vfmacc.vv    v24,   v4, v8;\n"
            "vfmacc.vv    v26,   v4, v10;\n"
            "vfmacc.vv    v28,   v4, v12;\n"
            "vfmacc.vv    v30,   v4, v14;\n"
            :
            : "r"(col0), "r"(col1), "r"(col2), "r"(col3), "r"(k0), "r"(k1), "r"(k2), "r"(k3));

        col0 += 32;
        col1 += 32;
        col2 += 32;
        col3 += 32;

        k0 += 32;
        k1 += 32;
        k2 += 32;
        k3 += 32;
    }

    for (; k < kernel_size; ++k)
    {
        asm("vle32.v        v0, (%0);\n"
            "vle32.v        v2, (%1);\n"

            "vrgather.vi    v8,  v2, 0;\n"
            "vrgather.vi    v10, v2, 1;\n"
            "vrgather.vi    v12, v2, 2;\n"
            "vrgather.vi    v14, v2, 3;\n"

            "vfmacc.vv      v16, v0, v8;\n"
            "vfmacc.vv      v18, v0, v10;\n"
            "vfmacc.vv      v20, v0, v12;\n"
            "vfmacc.vv      v22, v0, v14;\n"

            "vrgather.vi    v8,  v2, 4;\n"
            "vrgather.vi    v10, v2, 5;\n"
            "vrgather.vi    v12, v2, 6;\n"
            "vrgather.vi    v14, v2, 7;\n"

            "vfmacc.vv      v24, v0, v8;\n"
            "vfmacc.vv      v26, v0, v10;\n"
            "vfmacc.vv      v28, v0, v12;\n"
            "vfmacc.vv      v30, v0, v14;\n"
            :
            : "r"(col0), "r"(k0));
        col0 += 8;
        k0 += 8;
    }

    if (act >= 0)
    {
        asm(
            "vmv.v.x    v0, x0;\n"
            "vfmax.vv  v16, v16, v0;\n"
            "vfmax.vv  v18, v18, v0;\n"
            "vfmax.vv  v20, v20, v0;\n"
            "vfmax.vv  v22, v22, v0;\n"
            "vfmax.vv  v24, v24, v0;\n"
            "vfmax.vv  v26, v26, v0;\n"
            "vfmax.vv  v28, v28, v0;\n"
            "vfmax.vv  v30, v30, v0;\n");

        if (act > 0)
        {
            asm(
                "vmv.v.x    v2, %0;\n"
                "vfmin.vv  v16, v16, v2;\n"
                "vfmin.vv  v18, v18, v2;\n"
                "vfmin.vv  v20, v20, v2;\n"
                "vfmin.vv  v22, v22, v2;\n"
                "vfmin.vv  v24, v24, v2;\n"
                "vfmin.vv  v26, v26, v2;\n"
                "vfmin.vv  v28, v28, v2;\n"
                "vfmin.vv  v30, v30, v2;\n"
                :
                : "r"(act));
        }
    }

    float* r0 = cur_output;
    float* r1 = r0 + output_xy;
    float* r2 = r1 + output_xy;
    float* r3 = r2 + output_xy;
    float* r4 = r3 + output_xy;
    float* r5 = r4 + output_xy;
    float* r6 = r5 + output_xy;
    float* r7 = r6 + output_xy;

    switch (n)
    {
    case 8:
        asm(
            "vse32.v        v16, (%0);\n"
            "vse32.v        v18, (%1);\n"
            "vse32.v        v20, (%2);\n"
            "vse32.v        v22, (%3);\n"
            "vse32.v        v24, (%4);\n"
            "vse32.v        v26, (%5);\n"
            "vse32.v        v28, (%6);\n"
            "vse32.v        v30, (%7);\n"
            :
            : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r5), "r"(r6), "r"(r7));
        break;
    case 7:
        asm(
            "vse32.v        v16, (%0);\n"
            "vse32.v        v18, (%1);\n"
            "vse32.v        v20, (%2);\n"
            "vse32.v        v22, (%3);\n"
            "vse32.v        v24, (%4);\n"
            "vse32.v        v26, (%5);\n"
            "vse32.v        v28, (%6);\n"
            :
            : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r5), "r"(r6));
        break;

    case 6:
        asm(
            "vse32.v        v16, (%0);\n"
            "vse32.v        v18, (%1);\n"
            "vse32.v        v20, (%2);\n"
            "vse32.v        v22, (%3);\n"
            "vse32.v        v24, (%4);\n"
            "vse32.v        v26, (%5);\n"
            :
            : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r5));
        break;

    case 5:
        asm(
            "vse32.v        v16, (%0);\n"
            "vse32.v        v18, (%1);\n"
            "vse32.v        v20, (%2);\n"
            "vse32.v        v22, (%3);\n"
            "vse32.v        v24, (%4);\n"
            :
            : "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(r4));
        break;

    case 4:
        asm(
            "vse32.v        v16, (%0);\n"
            "vse32.v        v18, (%1);\n"
            "vse32.v        v20, (%2);\n"
            "vse32.v        v22, (%3);\n"
            :
            : "r"(r0), "r"(r1), "r"(r2), "r"(r3));
        break;

    case 3:
        asm(
            "vse32.v        v16, (%0);\n"
            "vse32.v        v18, (%1);\n"
            "vse32.v        v20, (%2);\n"
            :
            : "r"(r0), "r"(r1), "r"(r2));
        break;

    case 2:
        asm(
            "vse32.v        v16, (%0);\n"
            "vse32.v        v18, (%1);\n"
            :
            : "r"(r0), "r"(r1));
        break;

    case 1:
        asm(
            "vse32.v        v16, (%0);\n"
            :
            : "r"(r0));
        break;
    default:
        break;
    }
}
