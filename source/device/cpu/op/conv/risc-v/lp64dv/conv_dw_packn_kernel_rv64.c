#include "api/c_api.h"
#include <string.h>
#include "graph/graph.h"
#include "graph/node.h"
#include "graph/tensor.h"
#include "device/cpu/cpu_node.h"
#include "device/cpu/cpu_graph.h"
#include "device/cpu/cpu_module.h"
#include "op/conv/risc-v/lp64dv/vsetvl_rvv.h"
#include "utility/sys_port.h"
#include <stdio.h>
#include "utility/sys_port.h"
#include "convolution_param.h"

#define __likely(x)   __builtin_expect(!!(x), 1)
#define __unlikely(x) __builtin_expect(!!(x), 0)
#define max(a, b)     ((a) > (b) ? (a) : (b))
#define min(a, b)     ((a) < (b) ? (a) : (b))

// TODO: vectorize
static void pad(const float* input, float* output, const int in_h, const int in_w, const int out_h, const int out_w, const int top, const int left, const float v)
{
    float* ptr = input;
    float* outptr = output;

    int y = 0;
    // fill top
    for (; y < top; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
    // fill center
    for (; y < (top + in_h); y++)
    {
        int x = 0;
        for (; x < left; x++)
        {
            outptr[x] = v;
        }
        if (in_w < 12)
        {
            for (; x < (left + in_w); x++)
            {
                outptr[x] = ptr[x - left];
            }
        }
        else
        {
            memcpy(outptr + left, ptr, in_w * sizeof(float));
            x += in_w;
        }
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        ptr += in_w;
        outptr += out_w;
    }
    // fill bottom
    for (; y < out_h; y++)
    {
        int x = 0;
        for (; x < out_w; x++)
        {
            outptr[x] = v;
        }
        outptr += out_w;
    }
}

static void do_pack(const float* input, float* output, const int channels, const int feat_size, const int packn)
{
    const int channels_packed = (channels + packn - 1) / packn;
    const int feat_size_packed = feat_size * packn;
    const int input_num = channels * feat_size;

    int in = 0;

    for (int c = 0; c < channels_packed; ++c)
    {
        for (int i = 0; i < feat_size_packed; i += packn)
        {
            float* output_base = output + c * feat_size_packed + i;
            for (int k = 0; k < packn; ++k)
            {
                in = c * feat_size_packed + i / packn + k * feat_size;
                if (__likely(in < input_num))
                {
                    output_base[k] = input[in];
                }
                else
                {
                    output_base[k] = .0f;
                }
            }
        }
    }
}

// channels: packed_channels, feat_size: packed_feat_size
static void do_unpack(const float* packed, float* unpacked, const int packed_channels, const int packed_feat_size, const int unpacked_channels, const int packn)
{
    const int feat_size = packed_feat_size / packn;
    const int unpacked_num = unpacked_channels * packed_feat_size / packn;

    for (int c = 0; c < packed_channels; ++c)
    {
        for (int i = 0; i < packed_feat_size; i += packn)
        {
            const float* packed_base = packed + c * packed_feat_size + i;
            for (int k = 0; k < packn; ++k)
            {
                int out = c * packed_feat_size + i / packn + k * feat_size;
                if (__likely(out < unpacked_num))
                {
                    unpacked[out] = packed_base[k];
                }
            }
        }
    }
}

int conv_dw_packn_kernel_prerun(const ir_node_t* ir_node, const ir_tensor_t* input_tensor, const ir_tensor_t* filter_tensor, struct conv_priv_info* info, struct conv_param* params)
{
    const int inb = input_tensor->dims[0];
    const int inc = input_tensor->dims[1];
    const int inh = input_tensor->dims[2];
    const int inw = input_tensor->dims[3];

    const int pad_w = params->pad_w0;
    const int pad_h = params->pad_h0;
    const int inh_pad = inh + pad_h + pad_h;
    const int inw_pad = inw + pad_w + pad_w;

    if (inh_pad == inh && inw_pad == inw)
    {
        return 0;
    }

    if (!info->input_pad)
    {
        info->input_pad = sys_malloc(inb * inh_pad * inw_pad * inc * sizeof(float));
    }

    return 0;
}

int conv_dw_packn_kernel_postrun(const ir_node_t* ir_node, struct conv_priv_info* info)
{
    if (info->input_pad)
    {
        sys_free(info->input_pad);
    }

    return 0;
}

void convdw3x3s1_pack8_rvv(const float* input, const float* kernel, const float* bias, float* output, const int inc, const int inh, const int inw, const int outc, const int outh, const int outw, const int act, const struct conv_param* params, int num_thread)
{
    const int packn = 8;
    vsetvl_e32_m2();

#pragma omp parallel for num_threads(num_thread)
    for (int c = 0; c < inc; ++c)
    {
        const float* feat_map = input + c * inh * inw;
        const float* kernel_base = kernel + c * 9;
        const float* bias_base = bias ? bias + c : NULL;

        __asm__(
            "vle32.v     v18, (%0);\n"

            "vrgather.vi     v0,  v18, 0;\n"
            "vrgather.vi     v2,  v18, 1;\n"
            "vrgather.vi     v4,  v18, 2;\n"
            "vrgather.vi     v6,  v18, 3;\n"
            "vrgather.vi     v8,  v18, 4;\n"
            "vrgather.vi     v10, v18, 5;\n"
            "vrgather.vi     v12, v18, 6;\n"
            "vrgather.vi     v14, v18, 7;\n"

            "lw              t0, 32(%0);"
            "vmv.v.x     v16, t0;\n"
            :
            : "r"(kernel_base)
            : "t0");

        float* output_base = output + c * outw * outh;

        int h = 0;
        for (; h < (outh & -2); h += 2)
        {
            const float* row0 = feat_map + h * inw;
            const float* row1 = row0 + inw;
            const float* row2 = row1 + inw;
            const float* row3 = row2 + inw;

            int w = 0;
            for (; w < (outw & -packn); w += packn)
            {
                // bias = v18
                if (bias_base)
                {
                    __asm__("lw         t0, (%0)\n"
                            "vmv.v.x    v18, t0;\n"
                            "vmv.v.x    v20, t0;\n"
                            :
                            : "r"(bias_base)
                            : "t0");
                }
                else
                {
                    __asm__("vmv.v.x    v18, x0;\n"
                            "vmv.v.x    v20, x0;\n");
                }

                // r00, r01, r02, ..., r22 = v9, v10, v11, ...v17
                __asm__(
                    "vle32.v   v22, (%1);\n"
                    "addi       t0, %1, 4;\n"
                    "vle32.v   v24, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v26, (t0);\n"

                    "vfmacc.vv v18, v0, v22;\n"
                    "vfmacc.vv v18, v2, v24;\n"
                    "vfmacc.vv v18, v4, v26;\n"

                    "vle32.v   v22, (%2);\n"
                    "addi       t0, %2, 4;\n"
                    "vle32.v   v24, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v26, (t0);\n"

                    "vfmacc.vv v18, v6, v22;\n"
                    "vfmacc.vv v18, v8, v24;\n"
                    "vfmacc.vv v18, v10, v26;\n"

                    "vfmacc.vv v20, v0, v22;\n"
                    "vfmacc.vv v20, v2, v24;\n"
                    "vfmacc.vv v20, v4, v26;\n"

                    "vle32.v   v22, (%3);\n"
                    "addi       t0, %3, 4;\n"
                    "vle32.v   v24, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v26, (t0);\n"

                    "vfmacc.vv v18, v12, v22;\n"
                    "vfmacc.vv v18, v14, v24;\n"
                    "vfmacc.vv v18, v16, v26;\n"

                    "vfmacc.vv v20, v6, v22;\n"
                    "vfmacc.vv v20, v8, v24;\n"
                    "vfmacc.vv v20, v10, v26;\n"

                    "vle32.v   v22, (%4);\n"
                    "addi       t0, %4, 4;\n"
                    "vle32.v   v24, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v26, (t0);\n"

                    "vfmacc.vv v20, v12, v22;\n"
                    "vfmacc.vv v20, v14, v24;\n"
                    "vfmacc.vv v20, v16, v26;\n"
                    :
                    : "r"(output_base), "r"(row0), "r"(row1), "r"(row2), "r"(row3)
                    : "t0");

                if (act == 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vfmax.vv   v18, v18, v22;\n"
                            "vfmax.vv   v20, v20, v22;\n");
                }
                else if (act > 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vmv.v.x    v24, %0;\n"
                            "vfmax.vv   v18, v18, v22;\n"
                            "vfmin.vv   v18, v18, v24;\n"
                            "vfmax.vv   v20, v20, v22;\n"
                            "vfmin.vv   v20, v20, v24;\n"
                            :
                            : "r"(act));
                }

                __asm__("vse32.v    v18, (%0);\n" ::"r"(output_base));
                __asm__("vse32.v    v20, (%0);\n" ::"r"(output_base + outw));

                row0 += packn;
                row1 += packn;
                row2 += packn;
                row3 += packn;
                output_base += packn;
            }

            const float k00 = kernel_base[0];
            const float k01 = kernel_base[1];
            const float k02 = kernel_base[2];
            const float k10 = kernel_base[3];
            const float k11 = kernel_base[4];
            const float k12 = kernel_base[5];
            const float k20 = kernel_base[6];
            const float k21 = kernel_base[7];
            const float k22 = kernel_base[8];
            float bias_value = bias_base ? bias_base[0] : .0f;

            for (; w < outw; ++w)
            {
                const float i00 = row0[0];
                const float i01 = row0[1];
                const float i02 = row0[2];
                const float i10 = row1[0];
                const float i11 = row1[1];
                const float i12 = row1[2];
                const float i20 = row2[0];
                const float i21 = row2[1];
                const float i22 = row2[2];
                const float i30 = row3[0];
                const float i31 = row3[1];
                const float i32 = row3[2];

                float out1 = (k00 * i00 + k01 * i01 + k02 * i02 + k10 * i10 + k11 * i11 + k12 * i12 + k20 * i20 + k21 * i21 + k22 * i22 + bias_value);
                float out2 = (k00 * i10 + k01 * i11 + k02 * i12 + k10 * i20 + k11 * i21 + k12 * i22 + k20 * i30 + k21 * i31 + k22 * i32 + bias_value);

                if (act >= 0)
                {
                    out1 = max(out1, .0f);
                    out2 = max(out2, .0f);
                    if (act > 0)
                    {
                        out1 = min(out1, (float)act);
                        out2 = min(out2, (float)act);
                    }
                }

                *output_base = out1;
                *(output_base + outw) = out2;

                output_base += 1;
                row0 += 1;
                row1 += 1;
                row2 += 1;
                row3 += 1;
            }

            output_base += outw;
        }

        for (; h < outh; ++h)
        {
            const float* row0 = feat_map + h * inw;
            const float* row1 = row0 + inw;
            const float* row2 = row1 + inw;

            int w = 0;
            for (; w < (outw & -packn); w += packn)
            {
                // bias = v18
                if (bias_base)
                {
                    __asm__("lw         t0, (%0)\n"
                            "vmv.v.x    v18, t0;\n"
                            :
                            : "r"(bias_base)
                            : "t0");
                }
                else
                {
                    __asm__("vmv.v.x    v18, x0;\n");
                }

                // r00, r01, r02, ..., r22 = v9, v10, v11, ...v17
                __asm__(
                    "vle32.v   v22, (%0);\n"
                    "addi       t0, %0, 4;\n"
                    "vle32.v   v24, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v26, (t0);\n"

                    "vfmacc.vv v18, v0, v22;\n"
                    "vfmacc.vv v18, v2, v24;\n"
                    "vfmacc.vv v18, v4, v26;\n"

                    "vle32.v   v22, (%1);\n"
                    "addi       t0, %1, 4;\n"
                    "vle32.v   v24, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v26, (t0);\n"

                    "vfmacc.vv v18, v6, v22;\n"
                    "vfmacc.vv v18, v8, v24;\n"
                    "vfmacc.vv v18, v10, v26;\n"

                    "vle32.v   v22, (%2);\n"
                    "addi       t0, %2, 4;\n"
                    "vle32.v   v24, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v26, (t0);\n"

                    "vfmacc.vv v18, v12, v22;\n"
                    "vfmacc.vv v18, v14, v24;\n"
                    "vfmacc.vv v18, v16, v26;\n"
                    :
                    : "r"(row0), "r"(row1), "r"(row2)
                    : "t0");

                if (act == 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vfmax.vv   v18, v18, v22;\n");
                }
                else if (act > 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vmv.v.x    v24, %0;\n"
                            "vfmax.vv   v18, v18, v22;\n"
                            "vfmin.vv   v18, v18, v24;\n"
                            :
                            : "r"(act));
                }

                __asm__("vse32.v    v18, (%0);\n" ::"r"(output_base));

                row0 += packn;
                row1 += packn;
                row2 += packn;
                output_base += packn;
            }

            const float k00 = kernel_base[0];
            const float k01 = kernel_base[1];
            const float k02 = kernel_base[2];
            const float k10 = kernel_base[3];
            const float k11 = kernel_base[4];
            const float k12 = kernel_base[5];
            const float k20 = kernel_base[6];
            const float k21 = kernel_base[7];
            const float k22 = kernel_base[8];
            const float bias_value = bias_base ? bias_base[0] : .0f;

            for (; w < outw; ++w)
            {
                const float i00 = row0[0];
                const float i01 = row0[1];
                const float i02 = row0[2];
                const float i10 = row1[0];
                const float i11 = row1[1];
                const float i12 = row1[2];
                const float i20 = row2[0];
                const float i21 = row2[1];
                const float i22 = row2[2];

                float out1 = (k00 * i00 + k01 * i01 + k02 * i02 + k10 * i10 + k11 * i11 + k12 * i12 + k20 * i20 + k21 * i21 + k22 * i22 + bias_value);

                if (act >= 0)
                {
                    out1 = max(out1, .0f);
                    if (act > 0)
                    {
                        out1 = min(out1, (float)act);
                    }
                }

                *output_base = out1;

                output_base += 1;
                row0 += 1;
                row1 += 1;
                row2 += 1;
            }

            output_base += outw;
        }
    }
}

void convdw3x3s1_pack4_rvv(const float* input, const float* kernel, const float* bias, float* output, const int inc, const int inh, const int inw, const int outc, const int outh, const int outw, const int act, const struct conv_param* params, int num_thread)
{
    const int packn = 4;
    vsetvl_e32_m1();

#pragma omp parallel for num_threads(num_thread)
    for (int c = 0; c < inc; ++c)
    {
        const float* feat_map = input + c * inh * inw;
        const float* kernel_base = kernel + c * 9;
        const float* bias_base = bias ? bias + c : NULL;

        __asm__(
            "vle32.v     v9, (%0);\n"
            "addi        t0, %0, 16;\n"
            "vle32.v     v10, (t0);\n"

            "vrgather.vi     v0,  v9, 0;\n"
            "vrgather.vi     v1,  v9, 1;\n"
            "vrgather.vi     v2,  v9, 2;\n"
            "vrgather.vi     v3,  v9, 3;\n"
            "vrgather.vi     v4,  v10, 0;\n"
            "vrgather.vi     v5,  v10, 1;\n"
            "vrgather.vi     v6,  v10, 2;\n"
            "vrgather.vi     v7,  v10, 3;\n"

            "lw              t0, 32(%0);"
            "vmv.v.x     v8, t0;\n"
            :
            : "r"(kernel_base)
            : "t0");

        float* out0 = output + c * outw * outh;
        float* out1 = out0 + outw;
        float* out2 = out1 + outw;
        float* out3 = out2 + outw;

        int h = 0;
        for (; h < (outh & -4); h += 4)
        {
            const float* row0 = feat_map + h * inw;
            const float* row1 = row0 + inw;
            const float* row2 = row1 + inw;
            const float* row3 = row2 + inw;
            const float* row4 = row3 + inw;
            const float* row5 = row4 + inw;

            int w = 0;
            for (; w < (outw & -packn); w += packn)
            {
                // bias = v18
                if (bias_base)
                {
                    __asm__("lw         t0, (%0)\n"
                            "vmv.v.x    v28, t0;\n"
                            "vmv.v.x    v29, t0;\n"
                            "vmv.v.x    v30, t0;\n"
                            "vmv.v.x    v31, t0;\n"
                            :
                            : "r"(bias_base)
                            : "t0");
                }
                else
                {
                    __asm__("vmv.v.x    v28, x0;\n"
                            "vmv.v.x    v29, x0;\n"
                            "vmv.v.x    v30, x0;\n"
                            "vmv.v.x    v31, x0;\n");
                }

                // r00, r01, r02, ..., r22 = v9, v10, v11, ...v17
                __asm__(
                    "vle32.v    v9, (%0);\n"
                    "addi       t0, %0, 4;\n"
                    "vle32.v   v10, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v11, (t0);\n"

                    "vfmacc.vv v28, v0, v9;\n"
                    "vfmacc.vv v28, v1, v10;\n"
                    "vfmacc.vv v28, v2, v11;\n"

                    "vle32.v   v12, (%1);\n"
                    "addi       t0, %1, 4;\n"
                    "vle32.v   v13, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v14, (t0);\n"

                    "vfmacc.vv v28, v3, v12;\n"
                    "vfmacc.vv v28, v4, v13;\n"
                    "vfmacc.vv v28, v5, v14;\n"

                    "vfmacc.vv v29, v0, v12;\n"
                    "vfmacc.vv v29, v1, v13;\n"
                    "vfmacc.vv v29, v2, v14;\n"

                    "vle32.v   v15, (%2);\n"
                    "addi       t0, %2, 4;\n"
                    "vle32.v   v16, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v17, (t0);\n"

                    "vfmacc.vv  v28, v6, v15;\n"
                    "vfmacc.vv  v28, v7, v16;\n"
                    "vfmacc.vv  v28, v8, v17;\n"

                    "vfmacc.vv  v29, v3, v15;\n"
                    "vfmacc.vv  v29, v4, v16;\n"
                    "vfmacc.vv  v29, v5, v17;\n"

                    "vfmacc.vv  v30, v0, v15;\n"
                    "vfmacc.vv  v30, v1, v16;\n"
                    "vfmacc.vv  v30, v2, v17;\n"

                    "vle32.v   v18, (%3);\n"
                    "addi       t0, %3, 4;\n"
                    "vle32.v   v19, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v20, (t0);\n"

                    "vfmacc.vv v29, v6, v18;\n"
                    "vfmacc.vv v29, v7, v19;\n"
                    "vfmacc.vv v29, v8, v20;\n"

                    "vfmacc.vv v30, v3, v18;\n"
                    "vfmacc.vv v30, v4, v19;\n"
                    "vfmacc.vv v30, v5, v20;\n"

                    "vfmacc.vv v31, v0, v18;\n"
                    "vfmacc.vv v31, v1, v19;\n"
                    "vfmacc.vv v31, v2, v20;\n"

                    "vle32.v   v21, (%4);\n"
                    "addi       t0, %4, 4;\n"
                    "vle32.v   v22, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v23, (t0);\n"

                    "vfmacc.vv v30, v6, v21;\n"
                    "vfmacc.vv v30, v7, v22;\n"
                    "vfmacc.vv v30, v8, v23;\n"

                    "vfmacc.vv v31, v3, v21;\n"
                    "vfmacc.vv v31, v4, v22;\n"
                    "vfmacc.vv v31, v5, v23;\n"

                    "vle32.v   v24, (%5);\n"
                    "addi       t0, %5, 4;\n"
                    "vle32.v   v25, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v26, (t0);\n"

                    "vfmacc.vv v31, v6, v24;\n"
                    "vfmacc.vv v31, v7, v25;\n"
                    "vfmacc.vv v31, v8, v26;\n"
                    :
                    : "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(row4), "r"(row5)
                    : "t0");

                if (act == 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vfmax.vv   v28, v28, v22;\n"
                            "vfmax.vv   v29, v29, v22;\n"
                            "vfmax.vv   v30, v30, v22;\n"
                            "vfmax.vv   v31, v31, v22;\n");
                }
                else if (act > 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vmv.v.x    v23, %0;\n"
                            "vfmax.vv   v28, v28, v22;\n"
                            "vfmin.vv   v28, v28, v23;\n"
                            "vfmax.vv   v29, v29, v22;\n"
                            "vfmin.vv   v29, v29, v23;\n"
                            "vfmax.vv   v30, v30, v22;\n"
                            "vfmin.vv   v30, v30, v23;\n"
                            "vfmax.vv   v31, v31, v22;\n"
                            "vfmin.vv   v31, v31, v23;\n"
                            :
                            : "r"(act));
                }

                __asm__("vse32.v    v28, (%0);\n"
                        "vse32.v    v29, (%1);\n"
                        "vse32.v    v30, (%2);\n"
                        "vse32.v    v31, (%3);\n"
                        :
                        : "r"(out0), "r"(out1), "r"(out2), "r"(out3));

                row0 += packn;
                row1 += packn;
                row2 += packn;
                row3 += packn;
                row4 += packn;
                row5 += packn;

                out0 += packn;
                out1 += packn;
                out2 += packn;
                out3 += packn;
            }

            const float k00 = kernel_base[0];
            const float k01 = kernel_base[1];
            const float k02 = kernel_base[2];
            const float k10 = kernel_base[3];
            const float k11 = kernel_base[4];
            const float k12 = kernel_base[5];
            const float k20 = kernel_base[6];
            const float k21 = kernel_base[7];
            const float k22 = kernel_base[8];
            const float bias_value = bias_base ? bias_base[0] : .0f;

            for (; w < outw; ++w)
            {
                const float i00 = row0[0];
                const float i01 = row0[1];
                const float i02 = row0[2];

                const float i10 = row1[0];
                const float i11 = row1[1];
                const float i12 = row1[2];

                const float i20 = row2[0];
                const float i21 = row2[1];
                const float i22 = row2[2];

                const float i30 = row3[0];
                const float i31 = row3[1];
                const float i32 = row3[2];

                const float i40 = row4[0];
                const float i41 = row4[1];
                const float i42 = row4[2];

                const float i50 = row5[0];
                const float i51 = row5[1];
                const float i52 = row5[2];

                float v0 = (k00 * i00 + k01 * i01 + k02 * i02 + k10 * i10 + k11 * i11 + k12 * i12 + k20 * i20 + k21 * i21 + k22 * i22 + bias_value);
                float v1 = (k00 * i10 + k01 * i11 + k02 * i12 + k10 * i20 + k11 * i21 + k12 * i22 + k20 * i30 + k21 * i31 + k22 * i32 + bias_value);
                float v2 = (k00 * i20 + k01 * i21 + k02 * i22 + k10 * i30 + k11 * i31 + k12 * i32 + k20 * i40 + k21 * i41 + k22 * i42 + bias_value);
                float v3 = (k00 * i30 + k01 * i31 + k02 * i32 + k10 * i40 + k11 * i41 + k12 * i42 + k20 * i50 + k21 * i51 + k22 * i52 + bias_value);

                if (act >= 0)
                {
                    v0 = max(v0, .0f);
                    v1 = max(v1, .0f);
                    v2 = max(v2, .0f);
                    v3 = max(v3, .0f);

                    if (act > 0)
                    {
                        v0 = min(v0, (float)act);
                        v1 = min(v1, (float)act);
                        v2 = min(v2, (float)act);
                        v3 = min(v3, (float)act);
                    }
                }

                *out0 = v0;
                *out1 = v1;
                *out2 = v2;
                *out3 = v3;

                out0 += 1;
                out1 += 1;
                out2 += 1;
                out3 += 1;

                row0 += 1;
                row1 += 1;
                row2 += 1;
                row3 += 1;
                row4 += 1;
                row5 += 1;
            }

            out0 += 3 * outw;
            out1 += 3 * outw;
            out2 += 3 * outw;
            out3 += 3 * outw;
        }

        for (; h < outh; ++h)
        {
            const float* row0 = feat_map + h * inw;
            const float* row1 = row0 + inw;
            const float* row2 = row1 + inw;

            int w = 0;
            for (; w < (outw & -packn); w += packn)
            {
                // bias = v18
                if (bias_base)
                {
                    __asm__("lw         t0, (%0)\n"
                            "vmv.v.x    v28, t0;\n"
                            :
                            : "r"(bias_base)
                            : "t0");
                }
                else
                {
                    __asm__("vmv.v.x    v28, x0;\n");
                }

                // r00, r01, r02, ..., r22 = v9, v10, v11, ...v17
                __asm__(
                    "vle32.v    v9, (%0);\n"
                    "addi       t0, %0, 4;\n"
                    "vle32.v   v10, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v11, (t0);\n"

                    "vfmacc.vv v28, v0, v9;\n"
                    "vfmacc.vv v28, v1, v10;\n"
                    "vfmacc.vv v28, v2, v11;\n"

                    "vle32.v   v9, (%1);\n"
                    "addi       t0, %1, 4;\n"
                    "vle32.v   v10, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v11, (t0);\n"

                    "vfmacc.vv v28, v3, v9;\n"
                    "vfmacc.vv v28, v4, v10;\n"
                    "vfmacc.vv v28, v5, v11;\n"

                    "vle32.v   v9, (%2);\n"
                    "addi       t0, %2, 4;\n"
                    "vle32.v   v10, (t0);\n"
                    "addi       t0, t0, 4;\n"
                    "vle32.v   v11, (t0);\n"

                    "vfmacc.vv  v28, v6, v9;\n"
                    "vfmacc.vv  v28, v7, v10;\n"
                    "vfmacc.vv  v28, v8, v11;\n"
                    :
                    : "r"(row0), "r"(row1), "r"(row2)
                    : "t0");

                if (act == 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vfmax.vv   v28, v28, v22;\n");
                }
                else if (act > 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vmv.v.x    v23, %0;\n"
                            "vfmax.vv   v28, v28, v22;\n"
                            "vfmin.vv   v28, v28, v23;\n"
                            :
                            : "r"(act));
                }

                __asm__("vse32.v    v28, (%0);\n"
                        :
                        : "r"(out0));

                row0 += packn;
                row1 += packn;
                row2 += packn;

                out0 += packn;
            }

            const float k00 = kernel_base[0];
            const float k01 = kernel_base[1];
            const float k02 = kernel_base[2];
            const float k10 = kernel_base[3];
            const float k11 = kernel_base[4];
            const float k12 = kernel_base[5];
            const float k20 = kernel_base[6];
            const float k21 = kernel_base[7];
            const float k22 = kernel_base[8];
            const float bias_value = bias_base ? bias_base[0] : .0f;

            for (; w < outw; ++w)
            {
                const float i00 = row0[0];
                const float i01 = row0[1];
                const float i02 = row0[2];

                const float i10 = row1[0];
                const float i11 = row1[1];
                const float i12 = row1[2];

                const float i20 = row2[0];
                const float i21 = row2[1];
                const float i22 = row2[2];

                float v0 = (k00 * i00 + k01 * i01 + k02 * i02 + k10 * i10 + k11 * i11 + k12 * i12 + k20 * i20 + k21 * i21 + k22 * i22 + bias_value);

                if (act >= 0)
                {
                    v0 = max(v0, .0f);

                    if (act > 0)
                    {
                        v0 = min(v0, (float)act);
                    }
                }

                *out0 = v0;
                out0 += 1;

                row0 += 1;
                row1 += 1;
                row2 += 1;
            }
        }
    }
}

void convdw3x3s2_pack4_rvv(const float* input, const float* kernel, const float* bias, float* output, const int inc, const int inh, const int inw, const int outc, const int outh, const int outw, const int act, const struct conv_param* params, int num_thread)
{
    const int packn = 4;
    vsetvl_e32_m1();

#pragma omp parallel for num_threads(num_thread)
    for (int c = 0; c < inc; ++c)
    {
        const float* feat_map = input + c * inh * inw;
        const float* kernel_base = kernel + c * 9;
        const float* bias_base = bias ? bias + c : NULL;
        __asm__(
            "vle32.v     v9, (%0);\n"
            "addi        t0, %0, 16;\n"
            "vle32.v     v10, (t0);\n"

            "vrgather.vi     v0,  v9, 0;\n"
            "vrgather.vi     v1,  v9, 1;\n"
            "vrgather.vi     v2,  v9, 2;\n"
            "vrgather.vi     v3,  v9, 3;\n"
            "vrgather.vi     v4,  v10, 0;\n"
            "vrgather.vi     v5,  v10, 1;\n"
            "vrgather.vi     v6,  v10, 2;\n"
            "vrgather.vi     v7,  v10, 3;\n"

            "lw              t0, 32(%0);"
            "vmv.v.x     v8, t0;\n"
            :
            : "r"(kernel_base)
            : "t0");

        float* out0 = output + c * outw * outh;
        float* out1 = out0 + outw;
        float* out2 = out1 + outw;
        float* out3 = out2 + outw;

        int h = 0;
        for (; h < (outh & -4); h += 4)
        {
            const float* row0 = feat_map + 2 * h * inw;
            const float* row1 = row0 + inw;
            const float* row2 = row1 + inw;
            const float* row3 = row2 + inw;
            const float* row4 = row3 + inw;
            const float* row5 = row4 + inw;
            const float* row6 = row5 + inw;
            const float* row7 = row6 + inw;
            const float* row8 = row7 + inw;

            int w = 0;
            for (; w < (outw & -packn); w += packn)
            {
                // bias = v18
                if (bias_base)
                {
                    __asm__("lw         t0, (%0)\n"
                            "vmv.v.x    v28, t0;\n"
                            "vmv.v.x    v29, t0;\n"
                            "vmv.v.x    v30, t0;\n"
                            "vmv.v.x    v31, t0;\n"
                            :
                            : "r"(bias_base)
                            : "t0");
                }
                else
                {
                    __asm__("vmv.v.x    v28, x0;\n"
                            "vmv.v.x    v29, x0;\n"
                            "vmv.v.x    v30, x0;\n"
                            "vmv.v.x    v31, x0;\n");
                }

                // r00, r01, r02, ..., r22 = v9, v10, v11, ...v17
                __asm__(
                    "li         t1, 8;\n"
                    "vlse32.v   v9, (%0), t1;\n"
                    "addi       t0, %0, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v28, v0, v9;\n"
                    "vfmacc.vv v28, v1, v10;\n"
                    "vfmacc.vv v28, v2, v11;\n"

                    "vlse32.v   v9, (%1), t1;\n"
                    "addi       t0, %1, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v28, v3, v9;\n"
                    "vfmacc.vv v28, v4, v10;\n"
                    "vfmacc.vv v28, v5, v11;\n"

                    "vlse32.v   v9, (%2), t1;\n"
                    "addi       t0, %2, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v28, v6, v9;\n"
                    "vfmacc.vv v28, v7, v10;\n"
                    "vfmacc.vv v28, v8, v11;\n"

                    "vfmacc.vv v29, v0, v9;\n"
                    "vfmacc.vv v29, v1, v10;\n"
                    "vfmacc.vv v29, v2, v11;\n"

                    "vlse32.v   v9, (%3), t1;\n"
                    "addi       t0, %3, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v29, v3, v9;\n"
                    "vfmacc.vv v29, v4, v10;\n"
                    "vfmacc.vv v29, v5, v11;\n"

                    "vlse32.v   v9, (%4), t1;\n"
                    "addi       t0, %4, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v29, v6, v9;\n"
                    "vfmacc.vv v29, v7, v10;\n"
                    "vfmacc.vv v29, v8, v11;\n"

                    "vfmacc.vv v30, v0, v9;\n"
                    "vfmacc.vv v30, v1, v10;\n"
                    "vfmacc.vv v30, v2, v11;\n"

                    "vlse32.v   v9, (%5), t1;\n"
                    "addi       t0, %5, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v30, v3, v9;\n"
                    "vfmacc.vv v30, v4, v10;\n"
                    "vfmacc.vv v30, v5, v11;\n"

                    "vlse32.v   v9, (%6), t1;\n"
                    "addi       t0, %6, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v30, v6, v9;\n"
                    "vfmacc.vv v30, v7, v10;\n"
                    "vfmacc.vv v30, v8, v11;\n"

                    "vfmacc.vv v31, v0, v9;\n"
                    "vfmacc.vv v31, v1, v10;\n"
                    "vfmacc.vv v31, v2, v11;\n"

                    "vlse32.v   v9, (%7), t1;\n"
                    "addi       t0, %7, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v31, v3, v9;\n"
                    "vfmacc.vv v31, v4, v10;\n"
                    "vfmacc.vv v31, v5, v11;\n"

                    "vlse32.v   v9, (%8), t1;\n"
                    "addi       t0, %8, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v31, v6, v9;\n"
                    "vfmacc.vv v31, v7, v10;\n"
                    "vfmacc.vv v31, v8, v11;\n"
                    :
                    : "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(row4), "r"(row5), "r"(row6), "r"(row7), "r"(row8)
                    : "t0", "t1");

                if (act == 0)
                {
                    __asm__("vmv.v.x    v27, x0;\n"
                            "vfmax.vv   v28, v28, v27;\n"
                            "vfmax.vv   v29, v29, v27;\n"
                            "vfmax.vv   v30, v30, v27;\n"
                            "vfmax.vv   v31, v31, v27;\n");
                }
                else if (act > 0)
                {
                    __asm__("vmv.v.x    v26, x0;\n"
                            "vmv.v.x    v27, %0;\n"
                            "vfmax.vv   v28, v28, v26;\n"
                            "vfmin.vv   v28, v28, v27;\n"
                            "vfmax.vv   v29, v29, v26;\n"
                            "vfmin.vv   v29, v29, v27;\n"
                            "vfmax.vv   v30, v30, v26;\n"
                            "vfmin.vv   v30, v30, v27;\n"
                            "vfmax.vv   v31, v31, v26;\n"
                            "vfmin.vv   v31, v31, v27;\n"
                            :
                            : "r"(act));
                }

                __asm__(
                    "vse32.v    v28, (%0);\n"
                    "vse32.v    v29, (%1);\n"
                    "vse32.v    v30, (%2);\n"
                    "vse32.v    v31, (%3);\n"
                    :
                    : "r"(out0), "r"(out1), "r"(out2), "r"(out3));

                row0 += 2 * packn;
                row1 += 2 * packn;
                row2 += 2 * packn;
                row3 += 2 * packn;
                row4 += 2 * packn;
                row5 += 2 * packn;
                row6 += 2 * packn;
                row7 += 2 * packn;
                row8 += 2 * packn;
                out0 += packn;
                out1 += packn;
                out2 += packn;
                out3 += packn;
            }

            const float k00 = kernel_base[0];
            const float k01 = kernel_base[1];
            const float k02 = kernel_base[2];
            const float k10 = kernel_base[3];
            const float k11 = kernel_base[4];
            const float k12 = kernel_base[5];
            const float k20 = kernel_base[6];
            const float k21 = kernel_base[7];
            const float k22 = kernel_base[8];
            const float bias_value = bias_base ? bias_base[0] : .0f;

            for (; w < outw; ++w)
            {
                const float i00 = row0[0];
                const float i01 = row0[1];
                const float i02 = row0[2];
                const float i10 = row1[0];
                const float i11 = row1[1];
                const float i12 = row1[2];
                const float i20 = row2[0];
                const float i21 = row2[1];
                const float i22 = row2[2];
                const float i30 = row3[0];
                const float i31 = row3[1];
                const float i32 = row3[2];
                const float i40 = row4[0];
                const float i41 = row4[1];
                const float i42 = row4[2];
                const float i50 = row5[0];
                const float i51 = row5[1];
                const float i52 = row5[2];
                const float i60 = row6[0];
                const float i61 = row6[1];
                const float i62 = row6[2];
                const float i70 = row7[0];
                const float i71 = row7[1];
                const float i72 = row7[2];
                const float i80 = row8[0];
                const float i81 = row8[1];
                const float i82 = row8[2];

                float v0 = (k00 * i00 + k01 * i01 + k02 * i02 + k10 * i10 + k11 * i11 + k12 * i12 + k20 * i20 + k21 * i21 + k22 * i22 + bias_value);
                float v1 = (k00 * i20 + k01 * i21 + k02 * i22 + k10 * i30 + k11 * i31 + k12 * i32 + k20 * i40 + k21 * i41 + k22 * i42 + bias_value);
                float v2 = (k00 * i40 + k01 * i41 + k02 * i42 + k10 * i50 + k11 * i51 + k12 * i52 + k20 * i60 + k21 * i61 + k22 * i62 + bias_value);
                float v3 = (k00 * i60 + k01 * i61 + k02 * i62 + k10 * i70 + k11 * i71 + k12 * i72 + k20 * i80 + k21 * i81 + k22 * i82 + bias_value);

                if (act >= 0)
                {
                    v0 = max(v0, .0f);
                    v1 = max(v1, .0f);
                    v2 = max(v2, .0f);
                    v3 = max(v3, .0f);
                    if (act > 0)
                    {
                        v0 = min(v0, (float)act);
                        v1 = min(v1, (float)act);
                        v2 = min(v2, (float)act);
                        v3 = min(v3, (float)act);
                    }
                }

                *out0 = v0;
                *out1 = v1;
                *out2 = v2;
                *out3 = v3;

                out0 += 1;
                out1 += 1;
                out2 += 1;
                out3 += 1;

                row0 += 2;
                row1 += 2;
                row2 += 2;
                row3 += 2;
                row4 += 2;
                row5 += 2;
                row6 += 2;
                row7 += 2;
                row8 += 2;
            }

            out0 += 3 * outw;
            out1 += 3 * outw;
            out2 += 3 * outw;
            out3 += 3 * outw;
        }

        for (; h < outh; ++h)
        {
            const float* row0 = feat_map + 2 * h * inw;
            const float* row1 = row0 + inw;
            const float* row2 = row1 + inw;

            int w = 0;
            for (; w < (outw & -packn); w += packn)
            {
                // bias = v18
                if (bias_base)
                {
                    __asm__("lw         t0, (%0)\n"
                            "vmv.v.x    v28, t0;\n"
                            :
                            : "r"(bias_base)
                            : "t0");
                }
                else
                {
                    __asm__("vmv.v.x    v28, x0;\n");
                }

                // r00, r01, r02, ..., r22 = v9, v10, v11, ...v17
                __asm__(
                    "li         t1, 8;\n"
                    "vlse32.v   v9, (%0), t1;\n"
                    "addi       t0, %0, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v28, v0, v9;\n"
                    "vfmacc.vv v28, v1, v10;\n"
                    "vfmacc.vv v28, v2, v11;\n"

                    "vlse32.v   v9, (%1), t1;\n"
                    "addi       t0, %1, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v28, v3, v9;\n"
                    "vfmacc.vv v28, v4, v10;\n"
                    "vfmacc.vv v28, v5, v11;\n"

                    "vlse32.v   v9, (%2), t1;\n"
                    "addi       t0, %2, 4;\n"
                    "vlse32.v   v10, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v11, (t0), t1;\n"

                    "vfmacc.vv v28, v6, v9;\n"
                    "vfmacc.vv v28, v7, v10;\n"
                    "vfmacc.vv v28, v8, v11;\n"
                    :
                    : "r"(row0), "r"(row1), "r"(row2)
                    : "t0", "t1");

                if (act == 0)
                {
                    __asm__("vmv.v.x    v27, x0;\n"
                            "vfmax.vv   v28, v28, v27;\n");
                }
                else if (act > 0)
                {
                    __asm__("vmv.v.x    v26, x0;\n"
                            "vmv.v.x    v27, %0;\n"
                            "vfmax.vv   v28, v28, v26;\n"
                            "vfmin.vv   v28, v28, v27;\n"
                            :
                            : "r"(act));
                }

                __asm__(
                    "vse32.v    v28, (%0);\n"
                    :
                    : "r"(out0));

                row0 += 2 * packn;
                row1 += 2 * packn;
                row2 += 2 * packn;
                out0 += packn;
            }

            const float k00 = kernel_base[0];
            const float k01 = kernel_base[1];
            const float k02 = kernel_base[2];
            const float k10 = kernel_base[3];
            const float k11 = kernel_base[4];
            const float k12 = kernel_base[5];
            const float k20 = kernel_base[6];
            const float k21 = kernel_base[7];
            const float k22 = kernel_base[8];
            const float bias_value = bias_base ? bias_base[0] : .0f;

            for (; w < outw; ++w)
            {
                const float i00 = row0[0];
                const float i01 = row0[1];
                const float i02 = row0[2];
                const float i10 = row1[0];
                const float i11 = row1[1];
                const float i12 = row1[2];
                const float i20 = row2[0];
                const float i21 = row2[1];
                const float i22 = row2[2];

                float v0 = (k00 * i00 + k01 * i01 + k02 * i02 + k10 * i10 + k11 * i11 + k12 * i12 + k20 * i20 + k21 * i21 + k22 * i22 + bias_value);

                if (act >= 0)
                {
                    v0 = max(v0, .0f);
                    if (act > 0)
                    {
                        v0 = min(v0, (float)act);
                    }
                }

                *out0 = v0;

                out0 += 1;
                row0 += 2;
                row1 += 2;
                row2 += 2;
            }
        }
    }
}

void convdw3x3s2_pack8_rvv(const float* input, const float* kernel, const float* bias, float* output, const int inc, const int inh, const int inw, const int outc, const int outh, const int outw, const int act, const struct conv_param* params, int num_thread)
{
    const int packn = 8;

    vsetvl_e32_m2();
#pragma omp parallel for num_threads(num_thread)
    for (int c = 0; c < inc; ++c)
    {
        const float* feat_map = input + c * inh * inw;
        const float* kernel_base = kernel + c * 9;
        const float* bias_base = bias ? bias + c : NULL;

        __asm__(
            "vle32.v     v18, (%0);\n"

            "vrgather.vi     v0,  v18, 0;\n"
            "vrgather.vi     v2,  v18, 1;\n"
            "vrgather.vi     v4,  v18, 2;\n"
            "vrgather.vi     v6,  v18, 3;\n"
            "vrgather.vi     v8,  v18, 4;\n"
            "vrgather.vi     v10, v18, 5;\n"
            "vrgather.vi     v12, v18, 6;\n"
            "vrgather.vi     v14, v18, 7;\n"

            "lw              t0, 32(%0);"
            "vmv.v.x     v16, t0;\n"
            :
            : "r"(kernel_base));

        float* output_base = output + c * outw * outh;

        int h = 0;
        for (; h < (outh & -2); h += 2)
        {
            const float* row0 = feat_map + 2 * h * inw;
            const float* row1 = row0 + inw;
            const float* row2 = row1 + inw;
            const float* row3 = row2 + inw;
            const float* row4 = row3 + inw;

            int w = 0;
            for (; w < (outw & -packn); w += packn)
            {
                // bias = v18
                if (bias_base)
                {
                    __asm__("lw         t0, (%0)\n"
                            "vmv.v.x    v18, t0;\n"
                            "vmv.v.x    v20, t0;\n"
                            :
                            : "r"(bias_base)
                            : "t0");
                }
                else
                {
                    __asm__("vmv.v.x    v18, x0;\n"
                            "vmv.v.x    v20, x0;\n");
                }

                // r00, r01, r02, ..., r22 = v9, v10, v11, ...v17
                __asm__(
                    "li         t1, 8;\n"
                    "vlse32.v   v22, (%1), t1;\n"
                    "addi       t0, %1, 4;\n"
                    "vlse32.v   v24, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v26, (t0), t1;\n"

                    "vfmacc.vv v18, v0, v22;\n"
                    "vfmacc.vv v18, v2, v24;\n"
                    "vfmacc.vv v18, v4, v26;\n"

                    "vlse32.v   v22, (%2), t1;\n"
                    "addi       t0, %2, 4;\n"
                    "vlse32.v   v24, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v26, (t0), t1;\n"

                    "vfmacc.vv v18, v6, v22;\n"
                    "vfmacc.vv v18, v8, v24;\n"
                    "vfmacc.vv v18, v10, v26;\n"

                    "vlse32.v   v22, (%3), t1;\n"
                    "addi       t0, %3, 4;\n"
                    "vlse32.v   v24, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v26, (t0), t1;\n"

                    "vfmacc.vv v18, v12, v22;\n"
                    "vfmacc.vv v18, v14, v24;\n"
                    "vfmacc.vv v18, v16, v26;\n"

                    "vfmacc.vv v20, v0, v22;\n"
                    "vfmacc.vv v20, v2, v24;\n"
                    "vfmacc.vv v20, v4, v26;\n"

                    "vlse32.v   v22, (%4), t1;\n"
                    "addi       t0, %4, 4;\n"
                    "vlse32.v   v24, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v26, (t0), t1;\n"

                    "vfmacc.vv v20, v6, v22;\n"
                    "vfmacc.vv v20, v8, v24;\n"
                    "vfmacc.vv v20, v10, v26;\n"

                    "vlse32.v   v22, (%5), t1;\n"
                    "addi       t0, %5, 4;\n"
                    "vlse32.v   v24, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v26, (t0), t1;\n"

                    "vfmacc.vv v20, v12, v22;\n"
                    "vfmacc.vv v20, v14, v24;\n"
                    "vfmacc.vv v20, v16, v26;\n"
                    :
                    : "r"(output_base), "r"(row0), "r"(row1), "r"(row2), "r"(row3), "r"(row4)
                    : "t0", "t1");

                if (act == 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vfmax.vv   v18, v18, v22;\n"
                            "vfmax.vv   v20, v20, v22;\n");
                }
                else if (act > 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vmv.v.x    v24, %0;\n"
                            "vfmax.vv   v18, v18, v22;\n"
                            "vfmin.vv   v18, v18, v24;\n"
                            "vfmax.vv   v20, v20, v22;\n"
                            "vfmin.vv   v20, v20, v24;\n"
                            :
                            : "r"(act));
                }

                __asm__("vse32.v    v18, (%0);\n" ::"r"(output_base));
                __asm__("vse32.v    v20, (%0);\n" ::"r"(output_base + outw));

                row0 += 2 * packn;
                row1 += 2 * packn;
                row2 += 2 * packn;
                row3 += 2 * packn;
                row4 += 2 * packn;
                output_base += packn;
            }

            const float k00 = kernel_base[0];
            const float k01 = kernel_base[1];
            const float k02 = kernel_base[2];
            const float k10 = kernel_base[3];
            const float k11 = kernel_base[4];
            const float k12 = kernel_base[5];
            const float k20 = kernel_base[6];
            const float k21 = kernel_base[7];
            const float k22 = kernel_base[8];
            const float bias_value = bias_base ? bias_base[0] : .0f;

            for (; w < outw; ++w)
            {
                const float i00 = row0[0];
                const float i01 = row0[1];
                const float i02 = row0[2];
                const float i10 = row1[0];
                const float i11 = row1[1];
                const float i12 = row1[2];
                const float i20 = row2[0];
                const float i21 = row2[1];
                const float i22 = row2[2];
                const float i30 = row3[0];
                const float i31 = row3[1];
                const float i32 = row3[2];
                const float i40 = row4[0];
                const float i41 = row4[1];
                const float i42 = row4[2];

                float out1 = (k00 * i00 + k01 * i01 + k02 * i02 + k10 * i10 + k11 * i11 + k12 * i12 + k20 * i20 + k21 * i21 + k22 * i22 + bias_value);
                float out2 = (k00 * i20 + k01 * i21 + k02 * i22 + k10 * i30 + k11 * i31 + k12 * i32 + k20 * i40 + k21 * i41 + k22 * i42 + bias_value);

                if (act >= 0)
                {
                    out1 = max(out1, .0f);
                    out2 = max(out2, .0f);
                    if (act > 0)
                    {
                        out1 = min(out1, (float)act);
                        out2 = min(out2, (float)act);
                    }
                }

                *output_base = out1;
                *(output_base + outw) = out2;

                output_base += 1;
                row0 += 2;
                row1 += 2;
                row2 += 2;
                row3 += 2;
                row4 += 2;
            }

            output_base += outw;
        }

        for (; h < outh; ++h)
        {
            const float* row0 = feat_map + 2 * h * inw;
            const float* row1 = row0 + inw;
            const float* row2 = row1 + inw;

            int w = 0;
            for (; w < (outw & -packn); w += packn)
            {
                // bias = v18
                if (bias_base)
                {
                    __asm__("lw         t0, (%0)\n"
                            "vmv.v.x    v18, t0;\n"
                            :
                            : "r"(bias_base)
                            : "t0");
                }
                else
                {
                    __asm__("vmv.v.x    v18, x0;\n");
                }

                // r00, r01, r02, ..., r22 = v9, v10, v11, ...v17
                __asm__(
                    "li         t1, 8;\n"
                    "vlse32.v   v22, (%0), t1;\n"
                    "addi       t0, %0, 4;\n"
                    "vlse32.v   v24, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v26, (t0), t1;\n"

                    "vfmacc.vv v18, v0, v22;\n"
                    "vfmacc.vv v18, v2, v24;\n"
                    "vfmacc.vv v18, v4, v26;\n"

                    "vlse32.v   v22, (%1), t1;\n"
                    "addi       t0, %1, 4;\n"
                    "vlse32.v   v24, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v26, (t0), t1;\n"

                    "vfmacc.vv v18, v6, v22;\n"
                    "vfmacc.vv v18, v8, v24;\n"
                    "vfmacc.vv v18, v10, v26;\n"

                    "vlse32.v   v22, (%2), t1;\n"
                    "addi       t0, %2, 4;\n"
                    "vlse32.v   v24, (t0), t1;\n"
                    "addi       t0, t0, 4;\n"
                    "vlse32.v   v26, (t0), t1;\n"

                    "vfmacc.vv v18, v12, v22;\n"
                    "vfmacc.vv v18, v14, v24;\n"
                    "vfmacc.vv v18, v16, v26;\n"
                    :
                    : "r"(row0), "r"(row1), "r"(row2)
                    : "t0", "t1");

                if (act == 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vfmax.vv   v18, v18, v22;\n");
                }
                else if (act > 0)
                {
                    __asm__("vmv.v.x    v22, x0;\n"
                            "vfmax.vv   v18, v18, v22;\n"
                            "vfmin.vv   v18, v18, v24;\n"
                            :
                            : "r"(act));
                }

                __asm__("vse32.v    v18, (%0);\n" ::"r"(output_base));

                row0 += 2 * packn;
                row1 += 2 * packn;
                row2 += 2 * packn;
                output_base += packn;
            }

            const float k00 = kernel_base[0];
            const float k01 = kernel_base[1];
            const float k02 = kernel_base[2];
            const float k10 = kernel_base[3];
            const float k11 = kernel_base[4];
            const float k12 = kernel_base[5];
            const float k20 = kernel_base[6];
            const float k21 = kernel_base[7];
            const float k22 = kernel_base[8];
            const float bias_value = bias_base ? bias_base[0] : .0f;

            for (; w < outw; ++w)
            {
                const float i00 = row0[0];
                const float i01 = row0[1];
                const float i02 = row0[2];
                const float i10 = row1[0];
                const float i11 = row1[1];
                const float i12 = row1[2];
                const float i20 = row2[0];
                const float i21 = row2[1];
                const float i22 = row2[2];

                float out1 = (k00 * i00 + k01 * i01 + k02 * i02 + k10 * i10 + k11 * i11 + k12 * i12 + k20 * i20 + k21 * i21 + k22 * i22 + bias_value);

                if (act >= 0)
                {
                    out1 = max(out1, .0f);
                    if (act > 0)
                    {
                        out1 = min(out1, (float)act);
                    }
                }

                *output_base = out1;

                output_base += 1;
                row0 += 2;
                row1 += 2;
                row2 += 2;
            }
            output_base += outw;
        }
    }
}

int conv_dw_packn_kernel_run(const ir_node_t* ir_node, const ir_tensor_t* input_tensor, const ir_tensor_t* filter_tensor, const ir_tensor_t* bias_tensor, ir_tensor_t* output_tensor, const struct conv_priv_info* priv_info, const struct conv_param* params, const int num_thread, const int cpu_affinity)
{
    float* input = (float*)input_tensor->data;
    float* output = (float*)output_tensor->data;
    const float* kernel = filter_tensor->data;
    const float* bias = bias_tensor ? bias_tensor->data : NULL;

    const int inb = input_tensor->dims[0];
    const int inc = input_tensor->dims[1];
    const int inh = input_tensor->dims[2];
    const int inw = input_tensor->dims[3];

    const int outb = output_tensor->dims[0];
    const int outc = output_tensor->dims[1];
    const int outh = output_tensor->dims[2];
    const int outw = output_tensor->dims[3];

    const int ksize_h = params->kernel_h;
    const int ksize_w = params->kernel_w;
    const int pad_w = params->pad_w0;
    const int pad_h = params->pad_h0;
    const int stride_w = params->stride_w;
    const int stride_h = params->stride_h;

    const int dilation_w = params->dilation_w;
    const int dilation_h = params->dilation_h;
    const int group = params->group;
    const int act = params->activation;

    int inh_pad = inh + pad_h + pad_h;
    int inw_pad = inw + pad_w + pad_w;
    float* input_pad = NULL;

    if (inh_pad == inh && inw_pad == inw)
    {
        input_pad = input;
    }
    else
    {
        input_pad = priv_info->input_pad;
        for (int b = 0; b < inb; ++b)
        {
            const float* input_batch_base = input + b * inc * inh * inw;
            float* input_batch_padded_base = input_pad + b * inc * inh_pad * inw_pad;
#pragma omp parallel for num_threads(num_thread)
            for (int g = 0; g < group; ++g)
            {
                const float* pad_in = input_batch_base + g * inh * inw;
                float* pad_out = input_batch_padded_base + g * inh_pad * inw_pad;
                pad(pad_in, pad_out, inh, inw, inh_pad, inw_pad, pad_h, pad_w, .0f);
            }
        }
    }

    for (int b = 0; b < inb; ++b)
    {
        const float* input_batch_base = input_pad + b * inc * inh_pad * inw_pad;
        float* output_batch_base = output + b * outc * outh * outw;
        if (stride_h == 1)
        {
            convdw3x3s1_pack4_rvv(input_batch_base, kernel, bias, output_batch_base, inc, inh_pad, inw_pad, outc, outh, outw, act, params, num_thread);
        }
        else
        {
            convdw3x3s2_pack8_rvv(input_batch_base, kernel, bias, output_batch_base, inc, inh_pad, inw_pad, outc, outh, outw, act, params, num_thread);
        }
    }

    return 0;
}
