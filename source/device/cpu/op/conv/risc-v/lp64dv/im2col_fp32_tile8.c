#include <stdbool.h>
extern void im2col_fp32_1x1(const float* input, const int input_xy, const int input_chan, float* col);
extern void im2col_fp32_3x3(const float* input, int w, int h, int channel, float* cur_col, int stride);

static void trans_col(float* input, float* cur_col, int col_i, int in_c, int in_h, int in_w, int k_w, int k_h, int s_w, int s_h, int pad_w0, int pad_h0, int out_w, int out_h, int d_h, int d_w)
{
    const int in_xy = in_w * in_h;
    int cnt_y[] = {
        col_i / out_w,
        (col_i + 1) / out_w,
        (col_i + 2) / out_w,
        (col_i + 3) / out_w,
        (col_i + 4) / out_w,
        (col_i + 5) / out_w,
        (col_i + 6) / out_w,
        (col_i + 7) / out_w,
    };

    int cnt_x[] = {
        col_i - cnt_y[0] * out_w,
        col_i - cnt_y[1] * out_w + 1,
        col_i - cnt_y[2] * out_w + 2,
        col_i - cnt_y[3] * out_w + 3,
        col_i - cnt_y[4] * out_w + 4,
        col_i - cnt_y[5] * out_w + 5,
        col_i - cnt_y[6] * out_w + 6,
        col_i - cnt_y[7] * out_w + 7,
    };

    int imx_start[] = {
        cnt_x[0] * s_w - pad_w0,
        cnt_x[1] * s_w - pad_w0,
        cnt_x[2] * s_w - pad_w0,
        cnt_x[3] * s_w - pad_w0,
        cnt_x[4] * s_w - pad_w0,
        cnt_x[5] * s_w - pad_w0,
        cnt_x[6] * s_w - pad_w0,
        cnt_x[7] * s_w - pad_w0,
    };

    int imy_start[] = {
        cnt_y[0] * s_h - pad_h0,
        cnt_y[1] * s_h - pad_h0,
        cnt_y[2] * s_h - pad_h0,
        cnt_y[3] * s_h - pad_h0,
        cnt_y[4] * s_h - pad_h0,
        cnt_y[5] * s_h - pad_h0,
        cnt_y[6] * s_h - pad_h0,
        cnt_y[7] * s_h - pad_h0,
    };

    for (int kch = 0; kch < in_c; kch++)
    {
        for (int ky = 0; ky < (k_h * d_h); ky += d_h)
        {
            for (int kx = 0; kx < (k_w * d_w); kx += d_w)
            {
                int imx[8] = {
                    imx_start[0] + kx,
                    imx_start[1] + kx,
                    imx_start[2] + kx,
                    imx_start[3] + kx,
                    imx_start[4] + kx,
                    imx_start[5] + kx,
                    imx_start[6] + kx,
                    imx_start[7] + kx,
                };

                int imy[8] = {
                    imy_start[0] + ky,
                    imy_start[1] + ky,
                    imy_start[2] + ky,
                    imy_start[3] + ky,
                    imy_start[4] + ky,
                    imy_start[5] + ky,
                    imy_start[6] + ky,
                    imy_start[7] + ky,
                };

                for (int i = 0; i < 8; ++i)
                {
                    if (imx[i] >= 0 && imx[i] < in_w && imy[i] >= 0 && imy[i] < in_h)
                    {
                        *cur_col++ = *(input + in_xy * kch + in_w * imy[i] + imx[i]);
                    }
                    else
                    {
                        *cur_col++ = .0f;
                    }
                }
            }
        }
    }
}

void im2col(float* input, float* col, int in_c, int in_w, int in_h, int k_w, int k_h, int s_w, int s_h, int d_w,
            int d_h, int pad_w0, int pad_w1, int pad_h0, int pad_h1, int out_w, int out_h, int num_thread)
{
    const int kernel_size = k_w * k_h * in_c;
    const int in_xy = in_w * in_h;
    const int out_xy = out_w * out_h;
    const int col_end7 = out_xy & 7;
    const int is_pad0 = !(pad_h0 || pad_w0 || pad_h1 || pad_w1);

    if (k_w == 1 && k_h == 1 && s_w == 1 && s_h == 1)
    {
#pragma omp parallel for num_threads(num_thread)
        int col_i = 0;
        for (; col_i < (out_xy & -8); col_i += 8)
        {
            float* cur_col = col + col_i * kernel_size;

            int imy0 = col_i / out_w;
            int imy7 = (col_i + 7) / out_w;
            int imx0 = col_i - imy0 * out_w;
            int imx7 = (col_i + 7) - imy7 * out_w;

            int imx_start = imx0 * s_w - pad_w0;
            int imx_end = imx7 * s_w - pad_w0;
            int imy_start = imy0 * s_h - pad_h0;
            int imy_end = imy7 * s_h - pad_h0;

            // is pad ?
            if (imy0 == imy7 && (is_pad0 || (imx_start >= 0 && imx_end < in_w && imy_start >= 0 && imy_end < in_h)))
            {
                const float* cur_input = input + imy_start * in_w + imx_start;
                im2col_fp32_1x1(cur_input, in_xy, in_c, cur_col);
            }
            else
            {
                trans_col(input, cur_col, col_i, in_c, in_h, in_w, k_w, k_h, s_w, s_h, pad_w0, pad_h0, out_w, out_h, d_h, d_w);
            }
        }

        if (col_end7)
        {
            float* cur_col = col + col_i * kernel_size;
            trans_col(input, cur_col, col_i, in_c, in_h, in_w, k_w, k_h, s_w, s_h, pad_w0, pad_h0, out_w, out_h, d_h, d_w);
        }
    }
    else if (d_w == 1 && d_h == 1 && k_w == 3 && k_h == 3 && s_w == s_h)
    {
        int col_i = 0;
        for (; col_i < (out_xy & -8); col_i += 8)
        {
            float* cur_col = col + col_i * kernel_size;
            int imy0 = col_i / out_w;
            int imy7 = (col_i + 7) / out_w;
            int imx0 = col_i - imy0 * out_w;
            int imx7 = (col_i + 7) - imy7 * out_w;

            int imx_start = imx0 * s_w - pad_w0;
            int imx_end = imx7 * s_w - pad_w0;
            int imy_start = imy0 * s_h - pad_h0;
            int imy_end = imy7 * s_h - pad_h0;
            if ((imy0 == imy7) && (is_pad0 || (imx_start >= 0 && imx_end < in_w - 8 && imy_start >= 0 && imy_end + 2 < in_h)))
            {
                float* cur_input = input + imy_start * in_w + imx_start;
                im2col_fp32_3x3(cur_input, in_w, in_h, in_c, cur_col, s_w);
                cur_col += 8 * kernel_size;
            }
            else
            {
                trans_col(input, cur_col, col_i, in_c, in_h, in_w, k_w, k_h, s_w, s_h, pad_w0, pad_h0, out_w, out_h, d_h, d_w);
            }
        }

        if (col_end7)
        {
            float* cur_col = col + col_i * kernel_size;
            trans_col(input, cur_col, col_i, in_c, in_h, in_w, k_w, k_h, s_w, s_h, pad_w0, pad_h0, out_w, out_h, d_h, d_w);
        }
    }
    else
    {
        int col_i = 0;
        for (; col_i < (out_xy & -8); col_i += 8)
        {
            float* cur_col = col + col_i * kernel_size;
            trans_col(input, cur_col, col_i, in_c, in_h, in_w, k_w, k_h, s_w, s_h, pad_w0, pad_h0, out_w, out_h, d_h, d_w);
        }

        if (col_end7)
        {
            float* cur_col = col + col_i * kernel_size;
            trans_col(input, cur_col, col_i, in_c, in_h, in_w, k_w, k_h, s_w, s_h, pad_w0, pad_h0, out_w, out_h, d_h, d_w);
        }
    }
}
