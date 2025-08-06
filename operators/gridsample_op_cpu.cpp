#include <cstring> // For memcpy, memset
#include <string>
#include <cmath>   // For floor, round, fmin, fmax
#include <cstdlib>
#include <algorithm> // For std::max, std::min
#include <stdio.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h> // For NumPy C API

#include "rknn_custom_op.h"
#include "rknn_api.h"

// --- Helper Functions for Attribute Retrieval ---
// (Assuming these helper functions exist or are implemented similarly)
// These are conceptual, as the exact RKNN API for attribute access might differ slightly.
// You might need to adapt based on the actual RKNN SDK documentation.
static void dump_tensor_attr(rknn_tensor_attr* attr)
{
  std::string shape_str = attr->n_dims < 1 ? "" : std::to_string(attr->dims[0]);
  for (uint32_t i = 1; i < attr->n_dims; ++i) {
    shape_str += ", " + std::to_string(attr->dims[i]);
  }

  printf("  index=%d, name=%s, n_dims=%d, dims=[%s], n_elems=%d, size=%d, w_stride = %d, size_with_stride=%d, fmt=%s, "
         "type=%s, qnt_type=%s, "
         "zp=%d, scale=%f\n",
         attr->index, attr->name, attr->n_dims, shape_str.c_str(), attr->n_elems, attr->size, attr->w_stride,
         attr->size_with_stride, get_format_string(attr->fmt), get_type_string(attr->type),
         get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}
// --- Core Computation Logic ---

// Helper for bilinear sampling
void _bilinear_sample(
    const float* input_data, const float* grid_x_data, const float* grid_y_data,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    const char* padding_mode,
    float* output_data)
{
    // NCHW 格式：循环顺序应该是 N->C->H->W
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H_out; ++h) {
                for (int w = 0; w < W_out; ++w) {
                    float x = grid_x_data[n * H_out * W_out + h * W_out + w];
                    float y = grid_y_data[n * H_out * W_out + h * W_out + w];

                    // 计算双线性插值的四个邻近点
                    int x0 = static_cast<int>(std::floor(x));
                    int x1 = x0 + 1;
                    int y0 = static_cast<int>(std::floor(y));
                    int y1 = y0 + 1;

                    float wx = x - x0;
                    float wy = y - y0;

                    // 边界检查和处理
                    bool valid_x0 = (x0 >= 0) && (x0 < W_in);
                    bool valid_x1 = (x1 >= 0) && (x1 < W_in);
                    bool valid_y0 = (y0 >= 0) && (y0 < H_in);
                    bool valid_y1 = (y1 >= 0) && (y1 < H_in);

                    x0 = std::max(0, std::min(x0, W_in - 1));
                    x1 = std::max(0, std::min(x1, W_in - 1));
                    y0 = std::max(0, std::min(y0, H_in - 1));
                    y1 = std::max(0, std::min(y1, H_in - 1));

                    // NCHW 格式的索引计算
                    int out_idx = n * C * H_out * W_out + c * H_out * W_out + h * W_out + w;
                    
                    // 获取四个邻近点的值（NCHW格式）
                    float Q00 = input_data[n * C * H_in * W_in + c * H_in * W_in + y0 * W_in + x0];
                    float Q10 = input_data[n * C * H_in * W_in + c * H_in * W_in + y0 * W_in + x1];
                    float Q01 = input_data[n * C * H_in * W_in + c * H_in * W_in + y1 * W_in + x0];
                    float Q11 = input_data[n * C * H_in * W_in + c * H_in * W_in + y1 * W_in + x1];

                    // 处理边界
                    if (strcmp(padding_mode, "\"zeros\"") == 0) {
                        Q00 = valid_x0 && valid_y0 ? Q00 : 0.0f;
                        Q10 = valid_x1 && valid_y0 ? Q10 : 0.0f;
                        Q01 = valid_x0 && valid_y1 ? Q01 : 0.0f;
                        Q11 = valid_x1 && valid_y1 ? Q11 : 0.0f;
                    } else if(strcmp(padding_mode, "\"border\"") == 0){
                        // Border模式：使用clamp后的坐标，value已经是正确的
                    }

                    // 双线性插值
                    float Q0 = Q00 * (1.0f - wx) + Q10 * wx;
                    float Q1 = Q01 * (1.0f - wx) + Q11 * wx;
                    float result = Q0 * (1.0f - wy) + Q1 * wy;

                    output_data[out_idx] = result;
                }
            }
        }
    }
}

// Helper for nearest sampling
void _nearest_sample(
    const float* input_data, const float* grid_x_data, const float* grid_y_data,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    const char* padding_mode,
    float* output_data)
{
    // 如果是NHWC格式，应该是 N->H->W->C
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                // 修正：grid坐标访问
                float x = grid_x_data[n * H_out * W_out + h * W_out + w];
                float y = grid_y_data[n * H_out * W_out + h * W_out + w];

                int x_nearest = static_cast<int>(std::round(x));
                int y_nearest = static_cast<int>(std::round(y));

                bool valid_x = (x_nearest >= 0) && (x_nearest < W_in);
                bool valid_y = (y_nearest >= 0) && (y_nearest < H_in);

                // Clamp坐标到有效范围
                x_nearest = std::max(0, std::min(x_nearest, W_in - 1));
                y_nearest = std::max(0, std::min(y_nearest, H_in - 1));

                // 对每个通道进行采样
                for (int c = 0; c < C; ++c) {
                    // NHWC格式的索引计算
                    int out_idx = n * H_out * W_out * C + h * W_out * C + w * C + c;
                    int in_idx = n * H_in * W_in * C + y_nearest * W_in * C + x_nearest * C + c;

                    float value = input_data[in_idx];

                    // 处理边界填充
                    if (strcmp(padding_mode, "\"zeros\"") == 0) {
                        value = (valid_x && valid_y) ? value : 0.0f;
                    } else if (strcmp(padding_mode, "\"border\"") == 0) {
                        // Border模式：使用clamp后的坐标，value已经是正确的
                        // 无需额外处理
                    }

                    output_data[out_idx] = value;
                }
            }
        }
    }
}


// --- Main Custom Operator Compute Function ---
int compute_custom_grid_sample_float32(
    rknn_custom_op_context* op_ctx,
    rknn_custom_op_tensor* inputs, uint32_t n_inputs,
    rknn_custom_op_tensor* outputs, uint32_t n_outputs)
{
    // --- Input Validation ---
    if (n_inputs != 2 || n_outputs != 1) {
        // Incorrect number of inputs/outputs for GridSample
        PyErr_SetString(PyExc_RuntimeError, "Wrong Inputs");
        return -1; // Or appropriate error code
    }

    if (inputs[0].attr.type != RKNN_TENSOR_FLOAT32 ||
        inputs[1].attr.type != RKNN_TENSOR_FLOAT32 ||
        outputs[0].attr.type != RKNN_TENSOR_FLOAT32) {
        // Ensure float32 types
        return -1;
    }

    if (inputs[0].attr.n_dims != 4 || inputs[1].attr.n_dims != 4 || outputs[0].attr.n_dims != 4) {
         // Ensure 4D tensors
         return -1;
    }

    if (inputs[1].attr.dims[3] != 2) {
        // Grid last dimension must be 2
        return -1;
    }


    // --- Get Tensor Pointers ---
    float* input_data  = (float*)((unsigned char*)inputs[0].mem.virt_addr + inputs[0].mem.offset);
    float* grid_data   = (float*)((unsigned char*)inputs[1].mem.virt_addr + inputs[1].mem.offset);
    float* output_data = (float*)((unsigned char*)outputs[0].mem.virt_addr + outputs[0].mem.offset);

    // --- Get Tensor Shapes ---
    // Input: [N, C, H_in, W_in]
    int N_in = inputs[0].attr.dims[0];
    int C_in = inputs[0].attr.dims[1];
    int H_in = inputs[0].attr.dims[2];
    int W_in = inputs[0].attr.dims[3];

    // Grid: [N, H_out, W_out, 2]
    int N_grid = inputs[1].attr.dims[0];
    int H_out_grid = inputs[1].attr.dims[1];
    int W_out_grid = inputs[1].attr.dims[2];
    // int dim2_grid = inputs[1].attr.dims[3]; // Should be 2, checked above

    // Output: [N, C, H_out, W_out]
    int N_out = outputs[0].attr.dims[0];
    int C_out = outputs[0].attr.dims[1];
    int H_out = outputs[0].attr.dims[2];
    int W_out = outputs[0].attr.dims[3];

    // Shape consistency checks
    if (N_in != N_grid || N_in != N_out || C_in != C_out ||
        H_out != H_out_grid || W_out != W_out_grid) {
        return -1; // Mismatched dimensions
    }

    // --- Get Attributes using rknn_custom_op_get_op_attr ---
    rknn_custom_op_attr op_attr={};

    // Get 'mode' attribute (default "bilinear")
    char mode_str[32] = "\"bilinear\""; // Default value
    const char* mode = mode_str; // 默认值
    rknn_custom_op_get_op_attr(op_ctx, "mode", &op_attr);
    if (op_attr.data == NULL) {
        printf("Warning: Failed to get 'mode' attribute, using default 'bilinear'.\n");
        // Keep default value
    } else{
        mode = (const char*)op_attr.data; // Use the retrieved/default string
        // printf("%s\n",mode);fflush(stdout);
    }

    // Get 'padding_mode' attribute (default "zeros")
    char padding_mode_str[32] = "\"zeros\""; // Default value
    const char* padding_mode = padding_mode_str;
    op_attr = rknn_custom_op_attr{};
    rknn_custom_op_get_op_attr(op_ctx, "padding_mode", &op_attr);
    if (op_attr.data == NULL) {
        printf("Warning: Failed to get 'padding_mode' attribute, using default 'zeros'.\n");
        // Keep default value
    } else {
        padding_mode = (const char*)op_attr.data; // Use the retrieved/default string
        // printf("%s\n",padding_mode);fflush(stdout);
    }
    // Get 'align_corners' attribute (default false)
    bool align_corners = false; // Default value
    op_attr = rknn_custom_op_attr{};
    rknn_custom_op_get_op_attr(op_ctx, "align_corners", &op_attr);
    if (op_attr.data == NULL) {
        printf("Warning: Failed to get 'align_corners' attribute, using default false.\n");
        // Keep default value
    } else {
        bool tmp_bool = *(unsigned char*)op_attr.data;
        align_corners = tmp_bool;
        // printf("%d\n",align_corners);fflush(stdout);
    }

    // --- Compute Grid Coordinates ---
    // Allocate temporary buffers for transformed grid coordinates
    // In a more optimized version, you might avoid this by computing on the fly
    // or using a library that supports it.
    float* grid_x = (float*)malloc(N_out * H_out * W_out * sizeof(float));
    float* grid_y = (float*)malloc(N_out * H_out * W_out * sizeof(float));
    memset(grid_x, 0, N_out * H_out * W_out * sizeof(float));
    memset(grid_y, 0, N_out * H_out * W_out * sizeof(float));
    if (!grid_x || !grid_y) {
        free(grid_x); free(grid_y);
        return -1; // Allocation failed
    }

    // Transform grid coordinates from [-1, 1] to [0, W-1] or [0, H-1] etc.
    for (int n = 0; n < N_out; ++n) {
        for (int h = 0; h < H_out; ++h) {
            for (int w = 0; w < W_out; ++w) {
                 int grid_idx = n * H_out * W_out * 2 + h * W_out * 2 + w * 2;
                 float norm_x = grid_data[grid_idx + 0]; // x in [-1, 1]
                 float norm_y = grid_data[grid_idx + 1]; // y in [-1, 1]

                 float x_coord, y_coord;
                 if (align_corners) {
                     // Align corners: [-1, 1] -> [0, W_in-1] and [0, H_in-1]
                     x_coord = (norm_x + 1.0f) * (W_in - 1) / 2.0f;
                     y_coord = (norm_y + 1.0f) * (H_in - 1) / 2.0f;
                    //  printf("[debug] With align corners\n");fflush(stdout);
                 } else {
                     // No align corners: [-1, 1] -> [-0.5, W_in-0.5] and [-0.5, H_in-0.5]
                     x_coord = ((norm_x + 1.0f) * W_in - 1.0f) / 2.0f;
                     y_coord = ((norm_y + 1.0f) * H_in - 1.0f) / 2.0f;
                    //  printf("[debug] No align corners\n");fflush(stdout);
                 }
                 grid_x[n * H_out * W_out + h * W_out + w] = x_coord;
                 grid_y[n * H_out * W_out + h * W_out + w] = y_coord;
            }
        }
    }
    // printf("[debug] done transform\n");fflush(stdout);
    // --- Perform Sampling ---
    if (strcmp(mode, "\"bilinear\"") == 0) {
        _bilinear_sample(input_data, grid_x, grid_y, N_in, C_in, H_in, W_in, H_out, W_out, padding_mode, output_data);
        // printf("[debug] done bilinear\n");fflush(stdout);
    } else if (strcmp(mode, "\"nearest\"") == 0) {
        _nearest_sample(input_data, grid_x, grid_y, N_in, C_in, H_in, W_in, H_out, W_out, padding_mode, output_data);
        // printf("[debug] done nearest\n");fflush(stdout);
    } else {
        // Unsupported mode
        // printf("[debug] Unsupported mode\n");fflush(stdout);
        free(grid_x); free(grid_y);
        return -1;
    }

    // --- Cleanup ---
    free(grid_x);
    free(grid_y);

    return 0; // Success
}

static rknn_context ctx = 0;
static int model_loaded = 0;

static int load_model_internal(const char* model_path) {
    if (model_loaded) return 0;

    int ret;
    FILE *fp = fopen(model_path, "rb");
    if(fp == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to open model file");
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    size_t model_len = (size_t)ftell(fp);
    rewind(fp);
    char *model_data = (char*)malloc(model_len);
    if(model_data == NULL) {
        fclose(fp);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory for model");
        return -1;
    }
    if(fread(model_data, 1, model_len, fp) != model_len) {
        free(model_data);
        fclose(fp);
        PyErr_SetString(PyExc_RuntimeError, "Failed to read model file");
        return -1;
    }
    fclose(fp);

    ret = rknn_init(&ctx, model_data, model_len, 0, NULL);
    free(model_data);
    if(ret < 0) {
        PyErr_Format(PyExc_RuntimeError, "rknn_init failed with code %d", ret);
        return -1;
    }

    // register cstGridSample op
    rknn_custom_op user_op[1];
    memset(user_op, 0, sizeof(rknn_custom_op));

    // Set the op_type to match the one used in your model
    strncpy(user_op[0].op_type, "cstGridSample", RKNN_MAX_NAME_LEN - 1);
    user_op[0].version = 1;
    user_op[0].target  = RKNN_TARGET_TYPE_CPU; // Or appropriate target
    user_op[0].compute = compute_custom_grid_sample_float32; // Pointer to your function

    ret = rknn_register_custom_ops(ctx, user_op, 1);
    if (ret < 0) {
        PyErr_Format(PyExc_RuntimeError, "rknn_register_custom_op (cstGridSample) failed! ret = %d", ret);
        return -1;
    }
    printf("Custom op 'cstGridSample' registered successfully.\n");
    model_loaded = 1;

    printf("Model loaded in C extension.\n");
    return 0;
}

// The actual inference function exposed to Python
static PyObject* infer_rknn(PyObject* self, PyObject* args) {
    if (!model_loaded) {
        PyErr_SetString(PyExc_RuntimeError, "Model not loaded. Call load_model first.");
        return NULL;
    }

    PyArrayObject *input_array, *orig_size_array; // Expecting NumPy arrays
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &input_array, &PyArray_Type, &orig_size_array)) {
        // Note: This checks for PyArrayObject type, caller must pass numpy arrays
        return NULL; // Argument parsing failed
    }

    // --- Validate Input NumPy Arrays ---
    if (PyArray_NDIM(input_array) != 4 || PyArray_TYPE(input_array) != NPY_FLOAT32) {
         PyErr_SetString(PyExc_ValueError, "Input array must be 4D float32");
         return NULL;
    }
    if (PyArray_NDIM(orig_size_array) != 4 || PyArray_TYPE(orig_size_array) != NPY_INT64) {
         PyErr_SetString(PyExc_ValueError, "Orig size array must be 4D int64");
         return NULL;
    }

    rknn_input_output_num io_num;
    int ret;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return NULL;
    }
    // printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // printf("input tensors:\n");
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, io_num.n_input * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
        printf("rknn_query error! ret=%d\n", ret);
        return NULL;
        }
        // dump_tensor_attr(&input_attrs[i]);
    }

    // printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        // query info
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        return NULL;
        }
        // dump_tensor_attr(&output_attrs[i]);
    }

    // Get data pointers and shapes
    float *input_data = (float*)PyArray_DATA(input_array);
    npy_intp *input_shape = PyArray_DIMS(input_array);
    int64_t *orig_size_data = (int64_t*)PyArray_DATA(orig_size_array);
    npy_intp *orig_size_shape = PyArray_DIMS(orig_size_array);

    // --- Setup RKNN Inputs (similar to pure C example) ---
    rknn_input inputs[io_num.n_input];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].buf = input_data;
    inputs[0].size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * sizeof(float);
    inputs[0].pass_through = 0;
    inputs[0].type = RKNN_TENSOR_FLOAT32;
    inputs[0].fmt = input_attrs[0].fmt; // Adjust based on your model

    inputs[1].index = 1;
    inputs[1].buf = orig_size_data;
    inputs[1].size = orig_size_shape[0] * orig_size_shape[1] * orig_size_shape[2] * orig_size_shape[3] * sizeof(int64_t);
    inputs[1].pass_through = 0;
    inputs[1].type = RKNN_TENSOR_INT64;
    inputs[1].fmt = input_attrs[1].fmt; // Often doesn't matter for scalar/vector

    // --- Setup RKNN Outputs ---
    rknn_output outputs[io_num.n_output]; // Assuming 3 outputs
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 3; i++) {
        outputs[i].want_float = 1;
        outputs[i].is_prealloc = 0;
        outputs[i].index = i;
    }

    // --- Run Inference ---
    ret = rknn_inputs_set(ctx, 2, inputs);
    if (ret < 0) {
        PyErr_Format(PyExc_RuntimeError, "rknn_inputs_set failed with code %d", ret);
        return NULL;
    }
    // printf("[debug] done inputs set\n");fflush(stdout);
    ret = rknn_run(ctx, NULL);
    if (ret < 0) {
        PyErr_Format(PyExc_RuntimeError, "rknn_run failed with code %d", ret);
        return NULL;
    }
    // printf("[debug] done run\n");fflush(stdout);
    ret = rknn_outputs_get(ctx, 3, outputs, NULL);
    if (ret < 0) {
        PyErr_Format(PyExc_RuntimeError, "rknn_outputs_get failed with code %d", ret);
        return NULL;
    }
    // printf("[debug] done outputs set\n");fflush(stdout);
    // --- Convert RKNN outputs to NumPy arrays ---
    // Assuming outputs are [1, num_dets], [1, num_dets, 4], [1, num_dets]
    // We need to know the actual shape to create correct NumPy arrays.
    // This example assumes you know the output dims or get them from RKNN API
    // For simplicity, let's assume you query output attrs or know the shape
    // Let's say output 0 is labels [1, D], 1 is boxes [1, D, 4], 2 is scores [1, D]
    // We'll create 1D or 2D arrays for simplicity in Python (squeeze batch dim)
    // printf("[debug] dims_labels:%d\n", (int)(outputs[0].size / sizeof(float)));
    // printf("[debug] dims_boxes:%d\n", (int)(outputs[1].size / (4 * sizeof(float))));
    // printf("[debug] dims_scores:%d\n", (int)(outputs[2].size / sizeof(float)));
    fflush(stdout);
    npy_intp dims_labels[] = {(npy_intp)(outputs[0].size / sizeof(float))}; // [D]
    npy_intp dims_boxes[] = {(npy_intp)(outputs[1].size / (4 * sizeof(float))), 4}; // [D, 4]
    npy_intp dims_scores[] = {(npy_intp)(outputs[2].size / sizeof(float))}; // [D]

    // printf("[debug] 1\n");fflush(stdout);

    PyObject* py_labels = PyArray_SimpleNewFromData(1, dims_labels, NPY_FLOAT32, outputs[0].buf);
    PyObject* py_boxes = PyArray_SimpleNewFromData(2, dims_boxes, NPY_FLOAT32, outputs[1].buf);
    PyObject* py_scores = PyArray_SimpleNewFromData(1, dims_scores, NPY_FLOAT32, outputs[2].buf);

    // printf("[debug] 2\n");fflush(stdout);

    if (!py_labels || !py_boxes || !py_scores) {
        // Cleanup on error
        rknn_outputs_release(ctx, 3, outputs);
        Py_XDECREF(py_labels);
        Py_XDECREF(py_boxes);
        Py_XDECREF(py_scores);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create NumPy arrays for outputs");
        return NULL;
    }
    // Important: Tell NumPy it does not own the data, so it doesn't try to free it.
    // The data is managed by RKNN outputs. We need to ensure it's valid until Python uses it.
    // A better approach might be to copy the data or manage lifetime more carefully.
    // For now, we assume Python uses it immediately and we release it after.

    PyArray_CLEARFLAGS((PyArrayObject*)py_labels, NPY_ARRAY_OWNDATA);
    PyArray_CLEARFLAGS((PyArrayObject*)py_boxes, NPY_ARRAY_OWNDATA);
    PyArray_CLEARFLAGS((PyArrayObject*)py_scores, NPY_ARRAY_OWNDATA);

    // printf("[debug] 3\n");fflush(stdout);


    // Create a tuple to return the three arrays
    PyObject* result_tuple = PyTuple_New(3);
    // printf("[debug] tuple:%p\n", (void*)result_tuple);
    // printf("[debug] %d\n", !result_tuple);
    if (result_tuple) {
        // Better: Copy data into numpy arrays that Python owns, or manage lifetime properly.
        // For demonstration, let's assume we copy.
        // *** REVISION: Copy data to make NumPy arrays own it ***
        PyObject* py_labels_copy = PyArray_NewCopy((PyArrayObject*)py_labels, NPY_CORDER);
        PyObject* py_boxes_copy = PyArray_NewCopy((PyArrayObject*)py_boxes, NPY_CORDER);
        PyObject* py_scores_copy = PyArray_NewCopy((PyArrayObject*)py_scores, NPY_CORDER);
        Py_DECREF(py_labels);
        Py_DECREF(py_boxes);
        Py_DECREF(py_scores);
        py_labels = py_labels_copy;
        py_boxes = py_boxes_copy;
        py_scores = py_scores_copy;

        PyTuple_SetItem(result_tuple, 0, py_labels);
        PyTuple_SetItem(result_tuple, 1, py_boxes);
        PyTuple_SetItem(result_tuple, 2, py_scores);

        rknn_outputs_release(ctx, 3, outputs);
    } else {
        // If PyTuple_New failed, clean up numpy arrays
         PyErr_SetString(PyExc_RuntimeError, "Failed to create result tuple for outputs");
         Py_DECREF(py_labels);
         Py_DECREF(py_boxes);
         Py_DECREF(py_scores);
         rknn_outputs_release(ctx, 3, outputs);
         return NULL;
    }

    return result_tuple; // Return the tuple of NumPy arrays
}

// Function to load the model
static PyObject* load_model(PyObject* self, PyObject* args) {
    const char *model_path;
    if (!PyArg_ParseTuple(args, "s", &model_path)) {
        return NULL;
    }
    if (load_model_internal(model_path) < 0) {
        // Error already set by load_model_internal
        return NULL;
    }
    Py_RETURN_NONE; // Return Python None
}

static PyObject* release_model(PyObject* self, PyObject* args) {
    if (model_loaded && ctx != 0) {
        rknn_destroy(ctx);
        ctx = 0;
        model_loaded = 0;
        printf("Model context destroyed and resources released.\n");
    } else {
        printf("No model to release.\n");
    }
    Py_RETURN_NONE;
}

// Method definitions
static PyMethodDef module_methods[] = {
    {"load_model", load_model, METH_VARARGS, "Load RKNN model"},
    {"infer_rknn", infer_rknn, METH_VARARGS, "Run inference using RKNN C API"},
    {"release_model", release_model, METH_VARARGS, "Release RKNN model and free resources"},
    {NULL, NULL, 0, NULL} // Sentinel
};

// Module definition
static struct PyModuleDef rknninfermodule = {
    PyModuleDef_HEAD_INIT,
    "rknn_infer_dfine",   // Module name
    NULL,           // Module documentation
    -1,             // Size of per-interpreter state, -1 means global state
    module_methods  // Method definitions
};

// Module initialization function
PyMODINIT_FUNC PyInit_rknn_infer_dfine(void) {
    import_array(); // Required for NumPy C API
    if (PyErr_Occurred()) {
        return NULL;
    }
    return PyModule_Create(&rknninfermodule);
}