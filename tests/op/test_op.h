#ifndef __TEST_COMMON_H__
#define __TEST_COMMON_H__

#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>

//#include "float.h"
#include "api/c_api.h"
#include "tengine/c_api.h"
#include "mathp.h"
#include "vector.h"

#include "graph/graph.h"
#include "graph/subgraph.h"
#include "graph/node.h"
#include "graph/tensor.h"

#define TENSOR_SHOW_LEADING_BLANK "    "
#define TENSOR_FLOAT_EPSILON      0.0001f

struct data_buffer
{
    void* data;
    size_t size;
    int dims[8];
    int dim_num;
    int dtype;
    float scale;
    int32_t zero_point;
};

float random_float(float a, float b)
{
    float random = ((float)rand()) / (float)RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    float v = a + r;
    // generate denormal as zero
    if (v < 0.0001 && v > -0.0001)
        v = 0.f;
    return v;
}

int rand_int(const int a, const int b)
{
    const int delta = b - a;
    return a + rand() % delta;
}

struct data_buffer* create_data_buffer_from_tensor(tensor_t tensor)
{
    struct data_buffer* buf = (struct data_buffer*)malloc(sizeof(struct data_buffer));
    buf->size = get_tensor_buffer_size(tensor);
    buf->data = malloc(buf->size);
    memcpy(buf->data, get_tensor_buffer(tensor), buf->size);
    buf->dim_num = get_tensor_shape(tensor, buf->dims, 8);
    buf->dtype = get_tensor_data_type(tensor);
    get_tensor_quant_param(tensor, &buf->scale, &buf->zero_point, 1);
    return buf;
}

int dtype_to_size(const int dtype)
{
    switch (dtype)
    {
    case TENGINE_DT_FP32:
        return sizeof(float);
    case TENGINE_DT_INT8:
        return sizeof(int8_t);
    case TENGINE_DT_UINT8:
        return sizeof(uint8_t);
    case TENGINE_DT_FP16:
        return sizeof(uint16_t);
    case TENGINE_DT_INT16:
        return sizeof(int16_t);
    case TENGINE_DT_INT32:
        return sizeof(int32_t);
    default:
        return -1;
    }
}

struct data_buffer* create_data_buffer(const int* dims, const int dim_num, const int dtype)
{
    const int elem_size = dtype_to_size(dtype);
    if (elem_size < 0) return NULL;

    struct data_buffer* buf = (struct data_buffer*)malloc(sizeof(struct data_buffer));
    if (!buf) return NULL;
    buf->size = (int)(dim_num > 0);
    buf->dim_num = dim_num;

    for (int i = 0; i < dim_num; ++i)
    {
        buf->size *= dims[i];
        buf->dims[i] = dims[i];
    }

    buf->size *= elem_size;
    buf->dtype = dtype;
    buf->data = malloc(buf->size);
    if (!buf->data)
    {
        free(buf);
        return NULL;
    }

    buf->scale = random_float(-2.0, 2.0) + 0.01;
    buf->zero_point = rand_int(-10, 10);
    return buf;
}

struct data_buffer* create_data_buffer_fp32(const int* dims, const int dim_num)
{
    return create_data_buffer(dims, dim_num, TENGINE_DT_FP32);
}

void free_data_buffer_in_vector(void* p)
{
    struct data_buffer* buf = *(struct data_buffer**)p;
    free(buf->data);
    free(buf);
}

static float __fp16_to_fp32(uint16_t const value)
{
    union
    {
        struct
        {
            uint16_t frac : 10;
            uint16_t exp : 5;
            uint16_t sign : 1;
        } __attribute__((packed)) bits;

        uint16_t u16;
    } __attribute__((packed)) pack16 = {.u16 = value};

    union
    {
        struct
        {
            uint32_t frac : 23;
            uint32_t exp : 8;
            uint32_t sign : 1;
        } __attribute__((packed)) bits;
        uint32_t u32;
        float fp32;
    } __attribute__((packed)) pack32 = {.u32 = 0};

    if (pack16.bits.exp == 0 && pack16.bits.frac == 0)
    {
        pack32.u32 = 0;
        pack32.bits.sign = pack16.bits.sign;
        return pack32.fp32;
    }

    // normalized case
    if (pack16.bits.exp != 0xff && pack16.bits.exp != 0)
    {
        pack32.bits.sign = pack16.bits.sign;
        pack32.bits.exp = pack16.bits.exp - 15 + 127;
        pack32.bits.frac = pack16.bits.frac << 13;
        return pack32.fp32;
    }

    // subnormal case
    // 5.96046448e-8f = 2**-14 * 1/1024.0
    if (pack16.bits.exp == 0 && pack16.bits.frac != 0)
    {
        const float alpha = pack16.bits.sign == 0 ? 5.96046448e-8f : -5.96046448e-8f;
        return pack16.bits.frac * alpha;
    }

    if (pack16.bits.exp == 0x1f && pack16.bits.frac == 0)
    {
        pack32.bits.sign = pack16.bits.sign;
        pack32.bits.exp = 0xff;
        pack32.bits.frac = 0;
        return pack32.fp32;
    }

    if (pack16.bits.exp == 0x1f && pack16.bits.frac != 0)
    {
        pack32.bits.sign = pack16.bits.sign;
        pack32.bits.exp = 0xff;
        pack32.bits.frac = 1;
        return pack32.fp32;
    }

    return pack32.fp32;
}

bool is_match_buffer(const struct data_buffer* lhs, const struct data_buffer* rhs, const float eps)
{
    if (lhs->dim_num != rhs->dim_num || lhs->size != rhs->size || lhs->dtype != rhs->dtype) return false;
#define __compare(__dtype)                                                                                                                                                                                                                                   \
    do {                                                                                                                                                                                                                                                     \
        const __dtype* p1 = lhs->data;                                                                                                                                                                                                                       \
        const __dtype* p2 = rhs->data;                                                                                                                                                                                                                       \
        if (lhs->scale != rhs->scale || lhs->zero_point != rhs->zero_point) return false;                                                                                                                                                                    \
        for (int i = 0; i < lhs->size / dtype_to_size(lhs->dtype); ++i)                                                                                                                                                                                      \
        {                                                                                                                                                                                                                                                    \
            const int a = (int)p1[i];                                                                                                                                                                                                                        \
            const int b = (int)p2[i];                                                                                                                                                                                                                        \
            if (abs(a - b) != 0)                                                                                                                                                                                                                             \
            {                                                                                                                                                                                                                                                \
                fprintf(stderr, "buffer mismatch at %d, lhs = %d, rhs = %d, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", i, a, b, lhs->dims[0], lhs->dims[1], lhs->dims[2], lhs->dims[3], rhs->dims[0], rhs->dims[1], rhs->dims[2], rhs->dims[3]); \
                return false;                                                                                                                                                                                                                                \
            }                                                                                                                                                                                                                                                \
        }                                                                                                                                                                                                                                                    \
        return true;                                                                                                                                                                                                                                         \
    } while (0)

    for (int i = 0; i < lhs->dim_num; ++i)
    {
        if (lhs->dims[i] != rhs->dims[i]) return false;
    }

    if (lhs->dtype == TENGINE_DT_FP32)
    {
        const float* p1 = lhs->data;
        const float* p2 = rhs->data;

        for (int i = 0; i < lhs->size / sizeof(float); ++i)
        {
            if (fabs(p1[i] - p2[i]) > eps)
            {
                fprintf(stderr, "buffer mismatch at %d, lhs = %f, rhs = %f, dims1 = {%d, %d, %d, %d}, dims2 = {%d, %d, %d, %d}\n", i, p1[i], p2[i], lhs->dims[0], lhs->dims[1], lhs->dims[2], lhs->dims[3], rhs->dims[0], rhs->dims[1], rhs->dims[2], rhs->dims[3]);
                return false;
            }
        }

        return true;
    }
    else if (lhs->dtype == TENGINE_DT_UINT8)
    {
        __compare(uint8_t);
    }
    else if (lhs->dtype == TENGINE_DT_INT8)
    {
        __compare(int8_t);
    }
    else if (lhs->dtype == TENGINE_DT_INT32)
    {
        __compare(int32_t);
    }
    else if (lhs->dtype == TENGINE_DT_INT16)
    {
        __compare(int16_t);
    }
    else if (lhs->dtype == TENGINE_DT_FP16)
    {
        const uint16_t* p1 = lhs->data;
        const uint16_t* p2 = lhs->data;

        for (int i = 0; i < lhs->size; ++i)
        {
            const uint16_t a = p1[i];
            const uint16_t b = p2[i];
            const float fpa = __fp16_to_fp32(a);
            const float fpb = __fp16_to_fp32(b);

            if (fabs(fpa - fpb) > eps)
            {
                return false;
            }
        }

        return true;
    }
#undef __compare

    return false;
}

int fill_random_tensor(tensor_t v)
{
#define __fill(__dtype)                                            \
    do {                                                           \
        __dtype* p = get_tensor_buffer(v);                         \
        const int n = get_tensor_buffer_size(v) / sizeof(__dtype); \
        for (int i = 0; i < n; ++i)                                \
        {                                                          \
            if (dtype == TENGINE_DT_UINT8)                         \
            {                                                      \
                p[i] = (__dtype)rand_int(0, 30);                   \
            }                                                      \
            else                                                   \
            {                                                      \
                p[i] = (__dtype)rand_int(-15, 15);                 \
            }                                                      \
        }                                                          \
    } while (0);

    const int dtype = get_tensor_data_type(v);
    if (dtype == TENGINE_DT_FP32)
    {
        const int n = get_tensor_buffer_size(v);
        float* data = get_tensor_buffer(v);
        for (int i = 0; i < n / sizeof(float); ++i)
        {
            data[i] = random_float(-1.2, 1.2);
        }
        return 0;
    }
    else if (dtype == TENGINE_DT_INT8)
    {
        __fill(int8_t);
        return 0;
    }
    else if (dtype == TENGINE_DT_UINT8)
    {
        __fill(uint8_t);
        return 0;
    }
    else if (dtype == TENGINE_DT_INT32)
    {
        __fill(int32_t);
        return 0;
    }
    return -1;
}

typedef int (*node_setup_hook_fn)(graph_t graph, const char* test_node_name, const char* op, const char* input_name, int data_type, int input_num, int output_num);
typedef int (*common_test)(graph_t, const char* input_name, const char* node_name, int data_type, int layout, int n, int c, int h, int w);

#if 0
void dump_tensor_line(void* data_ptr, int offset, int data_type, int w)
{
    if (0 >= w)
    {
        fprintf(stderr, "Tensor.width = %d, not match width > 0.\n", w);
        return;
    }

    printf("[ ");

    switch (data_type)
    {
    case TENGINE_DT_FP32:
    {
        float* p = (float*)data_ptr;

        for (int i = 0; i < w - 1; i++)
        {
            printf("%0.2f, ", p[offset + i]);
        }
        printf("%0.2f ", p[offset + w - 1]);

        break;
    }
    case TENGINE_DT_FP16:
    {
        uint16_t* p = (uint16_t*)data_ptr;

#ifdef __ARM_ARCH
        for (int i = 0; i < w - 1; i++)
        {
            printf("%f, ", (float)p[offset + i]);
        }
        printf("%f ", (float)p[offset + w - 1]);
#else
        for (int i = 0; i < w - 1; i++)
        {
            printf("%f, ", fp16_to_fp32(p[offset + i]));
        }
        printf("%f ", fp16_to_fp32(p[offset + w - 1]));
#endif
        break;
    }
    case TENGINE_DT_INT8:
    case TENGINE_DT_UINT8:
    {
        if (data_type == TENGINE_DT_INT8)
        {
            int8_t* p = (int8_t*)data_ptr;

            for (int i = 0; i < w - 1; i++)
            {
                printf("%d, ", (int)p[offset + i]);
            }
            printf("%d ", (int)p[offset + w - 1]);
        }
        else
        {
            uint8_t* p = (uint8_t*)data_ptr;

            for (int i = 0; i < w - 1; i++)
            {
                printf("%d, ", (int)p[offset + i]);
            }
            printf("%d ", (int)p[offset + w - 1]);
        }

        break;
    }
    default:
        // not deal with TENGINE_DT_INT16 and TENGINE_DT_INT32
        fprintf(stderr, "Unsupported data type for now. ");
    }

    printf("]");
}

void dump_tensor(tensor_t tensor, const char* message)
{
    int data_type = get_tensor_data_type(tensor);
    void* data_ptr = get_tensor_buffer(tensor);

    int dim_array[MAX_SHAPE_DIM_NUM] = {0};
    int dim_count = get_tensor_shape(tensor, dim_array, MAX_SHAPE_DIM_NUM);
    if (0 >= dim_count)
        fprintf(stderr, "Cannot get tensor shape.");

    int line_count = 1;
    for (int i = 0; i < dim_count - 1; i++)
        line_count *= dim_array[i];

    int n = 0, c = 0, h = 0, w = 0;

    switch (dim_count)
    {
    case 4:
    {
        n = dim_array[0];
        c = dim_array[1];
        h = dim_array[2];
        w = dim_array[3];
        break;
    }
    case 3:
    {
        c = dim_array[0];
        h = dim_array[1];
        w = dim_array[2];
        break;
    }
    case 2:
    {
        h = dim_array[0];
        w = dim_array[1];
        break;
    }
    case 1:
    {
        w = dim_array[0];
        break;
    }
    default:
        fprintf(stderr, "Cannot found the type of tensor.\n");
    }

    // print leader
    printf("%s is { n, c, h, w } = { %d, %d, %d, %d }:\n", message, n, c, h, w);
    printf("[\n");

    for (int line = 0; line < line_count; line++)
    {
        if (2 <= dim_count && 0 == line % h)
            printf(TENSOR_SHOW_LEADING_BLANK "[\n");

        // print each line
        {
            for (int i = 0; i < dim_count - 2; i++)
                printf(TENSOR_SHOW_LEADING_BLANK);

            dump_tensor_line(data_ptr, line * w, data_type, w);

            if (0 != (line + 1) % h)
                printf(";\n");
            else
                printf("\n");
        }

        if (2 <= dim_count && 0 == (line + 1) % h)
        {
            if (line_count != line + 1)
                printf(TENSOR_SHOW_LEADING_BLANK "];\n");
            else
                printf(TENSOR_SHOW_LEADING_BLANK "]\n");
        }
    }
    printf("].\n");
}

void dump_node_input(node_t test_node, int index)
{
    tensor_t tensor = get_node_input_tensor(test_node, index);
    if (NULL == tensor)
    {
        fprintf(stderr, "Get input tensor(%d) from the node failed.\n", index);
        return;
    }

    char name[16] = {0};
    sprintf(name, "In%d", index);

    dump_tensor(tensor, name);

    release_graph_tensor(tensor);
}

void dump_node_output(node_t test_node, int index)
{
    tensor_t tensor = get_node_output_tensor(test_node, index);
    if (NULL == tensor)
    {
        fprintf(stderr, "Get output tensor from the node failed.\n");
        return;
    }

    char name[16] = {0};
    sprintf(name, "Out%d", index);

    dump_tensor(tensor, name);

    release_graph_tensor(tensor);
}
#endif

int create_node(graph_t graph, const char* node_name, int n, int c, int h, int w, int data_type, int layout)
{
    node_t node = create_graph_node(graph, node_name, "InputOp");
    if (NULL == node)
    {
        fprintf(stderr, "Create node(%s) with shape [n c h w] = [%d %d %d %d] failed.\n", node_name, n, c, h, w);
        return -1;
    }

    tensor_t tensor = create_graph_tensor(graph, node_name, data_type);
    if (NULL == tensor)
    {
        release_graph_node(node);

        fprintf(stderr, "Create tensor from node(%s) with shape [n c h w] = [%d %d %d %d] failed.\n", node_name, n, c, h, w);
        return -1;
    }

    set_node_output_tensor(node, 0, tensor, TENSOR_TYPE_INPUT);

    if (TENGINE_LAYOUT_NCHW == layout)
    {
        int dims[4] = {n, c, h, w};
        set_tensor_shape(tensor, dims, 4);
    }

    if (TENGINE_LAYOUT_NHWC == layout)
    {
        int dims[4] = {n, h, w, c};
        set_tensor_shape(tensor, dims, 4);
    }

    release_graph_tensor(tensor);
    release_graph_node(node);

    return 0;
}

int create_input_node_with_multi_inputs(graph_t graph, const char* node_name, int data_type, int input_num, int layout, int n, int c, int h, int w, int dims_count)
{
    if (0 == n) dims_count = 3;
    if (0 == c) dims_count = 2;
    if (0 == h) dims_count = 1;
    if (0 == w)
    {
        fprintf(stderr, "Dim of input node is not allowed. { n, c, h, w } = {%d, %d, %d, %d}.\n", n, c, h, w);
        return -1;
    }

    node_t node = create_graph_node(graph, node_name, OP_INPUT_NAME);
    if (NULL == node)
    {
        fprintf(stderr, "Create %d dims node(%s) failed. ", dims_count, node_name);
        return -1;
    }

    for (int i = 0; i < input_num; ++i)
    {
        char tensor_name[512];
        snprintf(tensor_name, sizeof(tensor_name), "%s_%d", node_name, i);
        tensor_t tensor = create_graph_tensor(graph, tensor_name, data_type);

        if (NULL == tensor)
        {
            release_graph_node(node);
            fprintf(stderr, "Create %d dims tensor for node(%s) failed. ", dims_count, node_name);
            return -1;
        }

        int ret = set_node_output_tensor(node, i, tensor, TENSOR_TYPE_INPUT);
        if (0 != ret)
        {
            release_graph_tensor(tensor);
            release_graph_node(node);
            fprintf(stderr, "Set %d dims output tensor for node(%s) failed. ", dims_count, node_name);
            return -1;
        }

        switch (dims_count)
        {
        case 1:
        {
            int dims_array[1] = {w};
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }
        case 2:
        {
            int dims_array[2] = {h, w};
            set_tensor_shape(tensor, dims_array, dims_count);
            break;
        }
        case 3:
        {
            if (TENGINE_LAYOUT_NCHW == layout)
            {
                int dims_array[3] = {c, h, w};
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }

            if (TENGINE_LAYOUT_NHWC == layout)
            {
                int dims_array[3] = {h, w, c};
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }
        }
        case 4:
        {
            if (TENGINE_LAYOUT_NCHW == layout)
            {
                int dims_array[4] = {n, c, h, w};
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }

            if (TENGINE_LAYOUT_NHWC == layout)
            {
                int dims_array[4] = {n, h, w, c};
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }
        }
        case 5:
        {
            if (TENGINE_LAYOUT_NCHW == layout)
            {
                int dims_array[5] = {1, n, c, h, w};
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }

            if (TENGINE_LAYOUT_NHWC == layout)
            {
                int dims_array[5] = {1, n, h, w, c};
                set_tensor_shape(tensor, dims_array, dims_count);
                break;
            }
        }
        default:
            fprintf(stderr, "Cannot support %d dims tensor.\n", dims_count);
        }
    }

    return 0;
}

int create_input_node(graph_t graph, const char* node_name, int data_type, int layout, int n, int c, int h, int w, int dims_count)
{
    return create_input_node_with_multi_inputs(graph, node_name, data_type, 1, layout, n, c, h, w, dims_count);
}

int fill_fp32_tensor(tensor_t tensor, float value)
{
    int dims[MAX_SHAPE_DIM_NUM];
    int dims_count = get_tensor_shape(tensor, dims, MAX_SHAPE_DIM_NUM);

    int type = get_tensor_data_type(tensor);

    if (TENGINE_DT_FP32 != type)
        return -1;

    int element_count = 1;
    for (int i = 0; i < dims_count; i++)
        element_count *= dims[i];

    if (0 == element_count)
        return -1;

    float* data_ptr = (float*)get_tensor_buffer(tensor);
    for (int i = 0; i < element_count; i++)
        data_ptr[i] = value;

    return 0;
}

int fill_int8_tensor(tensor_t tensor, float value)
{
    int dims[MAX_SHAPE_DIM_NUM];
    int dims_count = get_tensor_shape(tensor, dims, MAX_SHAPE_DIM_NUM);

    int type = get_tensor_data_type(tensor);

    if (TENGINE_DT_INT8 != type)
        return -1;

    int element_count = 1;
    for (int i = 0; i < dims_count; i++)
        element_count *= dims[i];

    if (0 == element_count)
        return -1;

    float input_scale = 0.f;
    int input_zero_point = 0;
    get_tensor_quant_param(tensor, &input_scale, &input_zero_point, 1);

    int8_t* data_ptr = (int8_t*)get_tensor_buffer(tensor);
    for (int i = 0; i < element_count; i++)
    {
        int data = (round)(value / input_scale + (float)input_zero_point);
        if (data > 127)
            data = 127;
        else if (data < -128)
            data = -128;
        data_ptr[i] = data;
    }

    return 0;
}

int fill_uint8_tensor(tensor_t tensor, float value)
{
    int dims[MAX_SHAPE_DIM_NUM];
    int dims_count = get_tensor_shape(tensor, dims, MAX_SHAPE_DIM_NUM);

    int type = get_tensor_data_type(tensor);

    if (TENGINE_DT_UINT8 != type)
        return -1;

    int element_count = 1;
    for (int i = 0; i < dims_count; i++)
        element_count *= dims[i];

    if (0 == element_count)
        return -1;

    float input_scale = 0.f;
    int input_zero_point = 0;
    get_tensor_quant_param(tensor, &input_scale, &input_zero_point, 1);

    uint8_t* data_ptr = (uint8_t*)get_tensor_buffer(tensor);
    for (int i = 0; i < element_count; i++)
    {
        int udata = (round)(value / input_scale + (float)input_zero_point);
        if (udata > 255)
            udata = 255;
        else if (udata < 0)
            udata = 0;
        data_ptr[i] = udata;
    }

    return 0;
}

void feed_input_tensor(graph_t graph, int input_node_idx, int input_tensor_idx, const float* values, int* dims, const int dim_num)
{
    tensor_t tensor = get_graph_input_tensor(graph, input_node_idx, input_tensor_idx);
    if (!tensor)
    {
        fprintf(stderr, "Cannot find %dth tensor with node idex %d\n", input_tensor_idx, input_node_idx);
        return;
    }
}

void fill_input_float_tensor_by_index(graph_t graph, int input_node_index, int tensor_index, float value)
{
    tensor_t tensor = get_graph_input_tensor(graph, input_node_index, tensor_index);
    if (NULL == tensor)
        fprintf(stderr, "Cannot find the %dth tensor via node index(%d).\n", tensor_index, input_node_index);

    int buf_size = get_tensor_buffer_size(tensor);
    float* data = (float*)malloc(buf_size);

    //    for(int i = 0; i < buf_size/sizeof(float); i++)
    //        data[i] = value;

    int ret = set_tensor_buffer(tensor, (void*)data, buf_size);
    if (0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");

    ret = fill_fp32_tensor(tensor, value);
    if (0 != ret)
        fprintf(stderr, "Fill buffer for tensor failed.\n");
}

void fill_input_int8_tensor_by_index(graph_t graph, int input_node_index, int tensor_index, float value)
{
    tensor_t tensor = get_graph_input_tensor(graph, input_node_index, tensor_index);
    if (NULL == tensor)
        fprintf(stderr, "Cannot find the %dth tensor via node index(%d).\n", tensor_index, input_node_index);

    int buf_size = get_tensor_buffer_size(tensor);
    int8_t* data = (int8_t*)malloc(buf_size);

    int ret = set_tensor_buffer(tensor, (void*)data, buf_size);
    if (0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");

    ret = fill_int8_tensor(tensor, value);
    if (0 != ret)
        fprintf(stderr, "Fill buffer for tensor failed.\n");
}

void fill_input_uint8_tensor_by_index(graph_t graph, int input_node_index, int tensor_index, float value)
{
    tensor_t tensor = get_graph_input_tensor(graph, input_node_index, tensor_index);
    if (NULL == tensor)
        fprintf(stderr, "Cannot find the %dth tensor via node index(%d).\n", tensor_index, input_node_index);

    int buf_size = get_tensor_buffer_size(tensor);
    uint8_t* data = (uint8_t*)malloc(buf_size);

    int ret = set_tensor_buffer(tensor, (void*)data, buf_size);
    if (0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");

    ret = fill_uint8_tensor(tensor, value);
    if (0 != ret)
        fprintf(stderr, "Fill buffer for tensor failed.\n");
}

void fill_input_float_tensor_by_name(graph_t graph, const char* node_name, int tensor_index, float value)
{
    node_t node = get_graph_node(graph, node_name);
    if (NULL == node)
        fprintf(stderr, "Cannot get node via node name(%s).\n", node_name);

    tensor_t tensor = get_node_input_tensor(node, tensor_index);
    if (NULL == tensor)
        fprintf(stderr, "Cannot find the %dth tensor via node name(%s)\n", tensor_index, node_name);

    int buf_size = get_tensor_buffer_size(tensor);
    float* data = (float*)malloc(buf_size);

    //    for(unsigned int i = 0; i < buf_size/sizeof(float) ; i++)
    //        data[i] = value;

    int ret = set_tensor_buffer(tensor, (void*)data, buf_size);
    if (0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");

    ret = fill_fp32_tensor(tensor, value);
    if (0 != ret)
        fprintf(stderr, "Fill buffer for tensor failed.\n");
}

void fill_input_float_buffer_tensor_by_name(graph_t graph, const char* node_name, int tensor_index, void* value, int buf_size)
{
    node_t node = get_graph_node(graph, node_name);
    if (NULL == node)
        fprintf(stderr, "Cannot get node via node name(%s).\n", node_name);

    tensor_t tensor = get_node_input_tensor(node, tensor_index);
    if (NULL == tensor)
        fprintf(stderr, "Cannot find the %dth tensor via node name(%s).\n", tensor_index, node_name);

    int ret = set_tensor_buffer(tensor, value, buf_size);
    if (0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");
}

void fill_input_integer_tensor_by_name(graph_t graph, const char* node_name, int tensor_index, int value)
{
    node_t node = get_graph_node(graph, node_name);
    if (NULL == node)
    {
        fprintf(stderr, "Cannot get node via node name(%s).\n", node_name);
        return;
    }

    tensor_t tensor = get_node_input_tensor(node, tensor_index);
    if (NULL == tensor)
    {
        fprintf(stderr, "Cannot find the %dth tensor via node name(%s).\n", tensor_index, node_name);
        return;
    }

    int buf_size = get_tensor_buffer_size(tensor);
    int* data = (int*)malloc(buf_size);

    for (unsigned int i = 0; i < buf_size / sizeof(int); i++)
        data[i] = value;

    int ret = set_tensor_buffer(tensor, (void*)data, buf_size);
    if (0 != ret)
        fprintf(stderr, "Set buffer for tensor failed.\n");
}

int test_graph_init()
{
    // now init tengine will mask critical filed and return an error
    // TODO: fix this fatal issue
    return init_tengine();
}

int test_graph_run(graph_t graph)
{
    if (prerun_graph(graph) < 0)
    {
        fprintf(stderr, "Pre-run graph failed.\n");
        return -1;
    }

    dump_graph(graph);

    if (0 != run_graph(graph, 1))
    {
        fprintf(stderr, "Run graph error.\n");
        return -1;
    }

    return 0;
}

void test_graph_release(graph_t graph)
{
    postrun_graph(graph);
    destroy_graph(graph);
}

static int craete_common_test_node(graph_t graph, const char* test_node_name, const char* op, const char* input_name, int data_type, int input_num, int output_num)
{
    node_t test_node = create_graph_node(graph, test_node_name, op);
    if (NULL == test_node)
    {
        fprintf(stderr, "create test node failed.\n");
        return -1;
    }

    node_t input_node = get_graph_node(graph, input_name);
    for (int i = 0; i < get_node_output_number(input_node); ++i)
    {
        tensor_t input_tensor = get_node_output_tensor(input_node, i);
        set_node_input_tensor(test_node, i, input_tensor);
    }

    char tensor_name[512];
    for (int i = 0; i < output_num; ++i)
    {
        snprintf(tensor_name, sizeof(tensor_name), "%s_%d", test_node_name, i);
        tensor_t output_tensor = create_graph_tensor(graph, tensor_name, data_type);
        if (!output_tensor)
        {
            fprintf(stderr, "create graph output tensor failed.\n");
            return -1;
        }

        set_node_output_tensor(test_node, i, output_tensor, TENSOR_TYPE_VAR);
    }
    return 0;
}

graph_t create_common_test_graph(const char* op, const char* test_node_name, const void* params, const size_t param_size, vector_t* inputs, int output_num, int data_type, int layout)
{
    graph_t graph = create_graph(NULL, NULL, NULL);
    if (NULL == graph)
    {
        fprintf(stderr, "get graph failed.\n");
        return NULL;
    }

    if (set_graph_layout(graph, layout) < 0)
    {
        fprintf(stderr, "set layout failed.\n");
        return NULL;
    }

    const char* input_name = "input_node";
    node_t input_node = create_graph_node(graph, input_name, OP_INPUT_NAME);
    node_t test_node = create_graph_node(graph, test_node_name, op);
    if (!input_node || !test_node)
    {
        fprintf(stderr, "create input node failed.\n");
        return NULL;
    }

    // setup input tensor
    char tensor_name[512];
    float scale = 1.0;
    int zero_point = 0.0;

    for (int i = 0; i < get_vector_num(inputs); ++i)
    {
        struct data_buffer* input = *(struct data_buffer**)get_vector_data(inputs, i);
        snprintf(tensor_name, sizeof(tensor_name), "%s_%d", input_name, i);
        tensor_t tensor = create_graph_tensor(graph, tensor_name, input->dtype);
        if (!tensor) return NULL;

        set_tensor_shape(tensor, input->dims, input->dim_num);
        set_tensor_buffer(tensor, input->data, input->size);
        scale = input->scale;
        zero_point = input->zero_point;
        set_tensor_quant_param(tensor, &scale, &zero_point, 1);

        if (set_node_output_tensor(input_node, i, tensor, TENSOR_TYPE_VAR))
        {
            return NULL;
        }

        if (set_node_input_tensor(test_node, i, tensor))
        {
            return NULL;
        }
    }

    // setup output tensor
    for (int i = 0; i < output_num; ++i)
    {
        snprintf(tensor_name, sizeof(tensor_name), "%s_%d", test_node_name, i);
        tensor_t output_tensor = create_graph_tensor(graph, tensor_name, data_type);

        if (data_type != TENGINE_DT_FP16 && data_type != TENGINE_DT_FP32)
        {
            set_tensor_quant_param(output_tensor, &scale, &zero_point, 1);
        }

        if (set_node_output_tensor(test_node, i, output_tensor, TENSOR_TYPE_VAR))
        {
            return NULL;
        }
    }

    // setup test node param
    if (params)
    {
        struct node* ir_node = (struct node*)test_node;
        memcpy(ir_node->op.param_mem, params, param_size);
    }

    // setup test node end.

    /* set input/output node */
    const char* input_nodes[] = {input_name};
    const char* output_nodes[] = {test_node_name};

    if (set_graph_input_node(graph, input_nodes, sizeof(input_nodes) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed.\n");
        return NULL;
    }

    if (set_graph_output_node(graph, output_nodes, sizeof(output_nodes) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed.\n");
        return NULL;
    }

    return graph;
}

//inputs: vector<struct data_buffer>
int create_common_op_test_case(const char* op, const void* params, const size_t param_size, vector_t* inputs, int output_num, int data_type, int layout, const float eps)
{
    int ret = test_graph_init();
    if (ret)
    {
        fprintf(stderr, "init test graph failed: %d\n", ret);
        return ret;
    }

    graph_t graph_ref = create_common_test_graph(op, "test_node", params, param_size, inputs, output_num, data_type, layout);

    vector_t* outputs_ref = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);

    for (int i = 0; i < get_graph_input_node_number(graph_ref); ++i)
    {
        node_t input_node = get_graph_input_node(graph_ref, i);
        for (int t = 0; t < get_node_output_number(input_node); ++t)
        {
            tensor_t input_tensor = get_graph_input_tensor(graph_ref, i, t);
            fill_random_tensor(input_tensor);
        }
    }

    setenv("TG_DEBUG_REF", "1", 1);

    if ((ret = test_graph_run(graph_ref)) < 0)
    {
        fprintf(stderr, "run graph failed: %d\n", ret);
        goto out;
    }

    for (int i = 0; i < get_graph_output_node_number(graph_ref); ++i)
    {
        node_t output_node = get_graph_output_node(graph_ref, i);
        for (int t = 0; t < get_node_output_number(output_node); ++t)
        {
            tensor_t output_tensor = get_graph_output_tensor(graph_ref, i, t);
            struct data_buffer* data = create_data_buffer_from_tensor(output_tensor);
            push_vector_data(outputs_ref, &data);
        }
    }
    test_graph_release(graph_ref);

    setenv("TG_DEBUG_REF", "0", 1);

    graph_t graph = create_common_test_graph(op, "test_node", params, param_size, inputs, output_num, data_type, layout);
    vector_t* outputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
    ret = test_graph_run(graph);
    if (ret)
    {
        fprintf(stderr, "run graph failed: %d\n", ret);
        goto out;
    }

    for (int i = 0; i < get_graph_output_node_number(graph); ++i)
    {
        node_t output_node = get_graph_output_node(graph, i);
        for (int t = 0; t < get_node_output_number(output_node); ++t)
        {
            tensor_t output_tensor = get_graph_output_tensor(graph, i, t);
            struct data_buffer* data = create_data_buffer_from_tensor(output_tensor);
            push_vector_data(outputs, &data);
        }
    }

    for (int i = 0; i < get_vector_num(outputs_ref); ++i)
    {
        struct data_buffer* p1 = *(struct data_buffer**)get_vector_data(outputs_ref, i);
        struct data_buffer* p2 = *(struct data_buffer**)get_vector_data(outputs, i);

        if (!is_match_buffer(p1, p2, eps))
        {
            fprintf(stderr, "%dth output is mismatch\n", i);
            ret = -1;
            goto out;
        }
    }

out:
    release_vector(outputs_ref);
    release_vector(outputs);
    test_graph_release(graph);
    release_tengine();
    return ret;
}

graph_t create_opendla_test_graph(const char* test_node_name, int data_type, int layout, int n, int c, int h, int w, common_test test_func, int dims_num)
{
    /* create OpenDLA backend */
    context_t odla_context = create_context("odla", 1);
    int rtt = set_context_device(odla_context, "OPENDLA", NULL, 0);
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        return NULL;
    }

    graph_t graph = create_graph(odla_context, NULL, NULL);
    if (NULL == graph)
    {
        fprintf(stderr, "get graph failed.\n");
        return NULL;
    }

    if (set_graph_layout(graph, layout) < 0)
    {
        fprintf(stderr, "set layout failed.\n");
        return NULL;
    }

    const char* input_name = "input_node";
    if (create_input_node(graph, input_name, data_type, layout, n, c, h, w, dims_num) < 0)
    {
        fprintf(stderr, "create input node failed.\n");
        return NULL;
    }

    if (test_func(graph, input_name, test_node_name, data_type, layout, n, c, h, w) < 0)
    {
        fprintf(stderr, "create test node failed.\n");
        return NULL;
    }

    /* set input/output node */
    const char* inputs[] = {input_name};
    const char* outputs[] = {test_node_name};

    if (set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed.\n");
        return NULL;
    }

    if (set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed.\n");
        return NULL;
    }

    return graph;
}

graph_t create_timvx_test_graph(const char* test_node_name, int data_type, int layout, int n, int c, int h, int w, common_test test_func, int dims_num)
{
    /* create VeriSilicon TIM-VX backend */
    context_t timvx_context = create_context("timvx", 1);
    int rtt = set_context_device(timvx_context, "TIMVX", NULL, 0);
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        return NULL;
    }

    graph_t graph = create_graph(timvx_context, NULL, NULL);
    if (NULL == graph)
    {
        fprintf(stderr, "get graph failed.\n");
        return NULL;
    }

    if (set_graph_layout(graph, layout) < 0)
    {
        fprintf(stderr, "set layout failed.\n");
        return NULL;
    }

    const char* input_name = "input_node";
    if (create_input_node(graph, input_name, data_type, layout, n, c, h, w, dims_num) < 0)
    {
        fprintf(stderr, "create input node failed.\n");
        return NULL;
    }

    if (test_func(graph, input_name, test_node_name, data_type, layout, n, c, h, w) < 0)
    {
        fprintf(stderr, "create test node failed.\n");
        return NULL;
    }

    /* set input/output node */
    const char* inputs[] = {input_name};
    const char* outputs[] = {test_node_name};

    if (set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed.\n");
        return NULL;
    }

    if (set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed.\n");
        return NULL;
    }

    return graph;
}

graph_t create_tensorrt_test_graph(const char* test_node_name, int data_type, int layout, int n, int c, int h, int w, common_test test_func, int dims_num)
{
    /* create TensorRT backend */
    context_t trt_context = create_context("tensorrt", 1);
    int rtt = set_context_device(trt_context, "TensorRT", NULL, 0);
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        return NULL;
    }

    graph_t graph = create_graph(trt_context, NULL, NULL);
    if (NULL == graph)
    {
        fprintf(stderr, "get graph failed.\n");
        return NULL;
    }

    if (set_graph_layout(graph, layout) < 0)
    {
        fprintf(stderr, "set layout failed.\n");
        return NULL;
    }

    const char* input_name = "input_node";
    if (create_input_node(graph, input_name, data_type, layout, n, c, h, w, dims_num) < 0)
    {
        fprintf(stderr, "create input node failed.\n");
        return NULL;
    }

    if (test_func(graph, input_name, test_node_name, data_type, layout, n, c, h, w) < 0)
    {
        fprintf(stderr, "create test node failed.\n");
        return NULL;
    }

    /* set input/output node */
    const char* inputs[] = {input_name};
    const char* outputs[] = {test_node_name};

    if (set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed.\n");
        return NULL;
    }

    if (set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed.\n");
        return NULL;
    }

    return graph;
}

graph_t create_torch_test_graph(const char* test_node_name, int data_type, int layout, int n, int c, int h, int w, common_test test_func, int dims_num)
{
    /* create libTorch backend */
    context_t torch_context = create_context("torch", 1);
    int rtt = set_context_device(torch_context, "TORCH", NULL, 0);
    if (0 > rtt)
    {
        fprintf(stderr, " add_context_device VSI DEVICE failed.\n");
        return NULL;
    }

    graph_t graph = create_graph(torch_context, NULL, NULL);
    if (NULL == graph)
    {
        fprintf(stderr, "get graph failed.\n");
        return NULL;
    }

    if (set_graph_layout(graph, layout) < 0)
    {
        fprintf(stderr, "set layout failed.\n");
        return NULL;
    }

    const char* input_name = "input_node";
    if (create_input_node(graph, input_name, data_type, layout, n, c, h, w, dims_num) < 0)
    {
        fprintf(stderr, "create input node failed.\n");
        return NULL;
    }

    if (test_func(graph, input_name, test_node_name, data_type, layout, n, c, h, w) < 0)
    {
        fprintf(stderr, "create test node failed.\n");
        return NULL;
    }

    /* set input/output node */
    const char* inputs[] = {input_name};
    const char* outputs[] = {test_node_name};

    if (set_graph_input_node(graph, inputs, sizeof(inputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set inputs failed.\n");
        return NULL;
    }

    if (set_graph_output_node(graph, outputs, sizeof(outputs) / sizeof(char*)) < 0)
    {
        fprintf(stderr, "set outputs failed.\n");
        return NULL;
    }

    return graph;
}

static inline unsigned long get_current_time(void)
{
    struct timespec tm;

    clock_gettime(CLOCK_MONOTONIC, &tm);

    return (tm.tv_sec * 1000000 + tm.tv_nsec / 1000);
}

#endif
