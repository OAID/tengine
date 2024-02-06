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
};

struct data_buffer* create_data_buffer(tensor_t tensor)
{
    struct data_buffer* buf = (struct data_buffer*)malloc(sizeof(struct data_buffer));
    buf->size = get_tensor_buffer_size(tensor);
    buf->data = malloc(buf->size);
    memcpy(buf->data, get_tensor_buffer(tensor), buf->size);
    return buf;
}

void free_data_buffer_in_vector(void* p)
{
    struct data_buffer* buf = *(struct data_buffer**)p;
    free(buf->data);
    free(buf);
}

bool is_match_buffer_fp32(const struct data_buffer* lhs, const struct data_buffer* rhs, const float eps)
{
    if (lhs->size != rhs->size) return false;
    float* p1 = lhs->data;
    float* p2 = rhs->data;

    for (int i = 0; i < lhs->size / sizeof(float); ++i)
    {
        if (fabs(p1[i] - p2[i]) > eps)
        {
            return false;
        }
    }

    return true;
}

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

void fill_random_tensor_fp32(tensor_t v)
{
    const int n = get_tensor_buffer_size(v);
    float* data = (float*)malloc(n);
    for (int i = 0; i < n / sizeof(float); ++i)
    {
        data[i] = random_float(-1.2, 1.2);
    }
    set_tensor_buffer(v, data, n);
}

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
    init_tengine();

    return 0;
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
    release_tengine();
}

graph_t create_common_test_graph(const char* op, const char* test_node_name, int data_type, int input_num, int output_num, int layout, int n, int c, int h, int w, int dims_num)
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
    if (create_input_node_with_multi_inputs(graph, input_name, data_type, input_num, layout, n, c, h, w, dims_num) < 0)
    {
        fprintf(stderr, "create input node failed.\n");
        return NULL;
    }

    // setup test node
    node_t test_node = create_graph_node(graph, test_node_name, op);
    if (NULL == test_node)
    {
        fprintf(stderr, "create test node failed.\n");
        return NULL;
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
            return NULL;
        }

        set_node_output_tensor(test_node, i, output_tensor, TENSOR_TYPE_VAR);
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

int create_common_op_test_case(const char* op, int input_num, int output_num, int data_type, int layout, const int* dims, int dims_num, const float eps)
{
    int n = 1, c = 1, h = 1, w = 1;
    switch (dims_num)
    {
    case 0:
        return -1;
    case 1: w = 1; break;
    case 2:
        h = dims[0];
        w = dims[1];
        break;
    case 3:
        c = dims[0];
        h = dims[1];
        w = dims[2];
        break;
    case 4:
        n = dims[0];
        c = dims[1];
        h = dims[2];
        w = dims[3];
        break;
    default:
        return -1;
    }

    int ret = test_graph_init();
    if (ret)
    {
        fprintf(stderr, "init test graph failed: %d\n", ret);
        return ret;
    }

    graph_t graph = create_common_test_graph(op, "test_node", data_type, input_num, output_num, layout, n, c, h, w, dims_num);
    vector_t* outputs_ref = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);
    vector_t* outputs = create_vector(sizeof(struct data_buffer*), free_data_buffer_in_vector);

    for (int i = 0; i < get_graph_input_node_number(graph); ++i)
    {
        node_t input_node = get_graph_input_node(graph, i);
        for (int t = 0; t < get_node_output_number(input_node); ++t)
        {
            tensor_t input_tensor = get_graph_input_tensor(graph, i, t);
            fill_random_tensor_fp32(input_tensor);
        }
    }

    setenv("TG_DEBUG_REF", "1", 1);
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
            struct data_buffer* data = create_data_buffer(output_tensor);
            push_vector_data(outputs_ref, &data);
        }
    }

    setenv("TG_DEBUG_REF", "0", 1);
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
            struct data_buffer* data = create_data_buffer(output_tensor);
            push_vector_data(outputs, &data);
        }
    }

    for (int i = 0; i < get_vector_num(outputs_ref); ++i)
    {
        struct data_buffer* p1 = get_vector_data(outputs_ref, i);
        struct data_buffer* p2 = get_vector_data(outputs, i);
        if (!is_match_buffer_fp32(p1, p2, eps))
        {
            fprintf(stderr, "%dth output is mismatch\n", i);
            ret = -1;
            goto out;
        }
    }

out:
    test_graph_release(graph);
    release_vector(outputs);
    release_vector(outputs_ref);
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

graph_t create_cpu_test_graph(const char* test_node_name, int data_type, int layout, int n, int c, int h, int w, common_test test_func, int dims_num)
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
