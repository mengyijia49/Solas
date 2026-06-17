#include "solas/solas.h"

#include <new>
#include <string>
#include <vector>
#include <cstddef>

struct SolasTensor {
    std::vector<long long> shape;//表示每个 Tensor 内部保存一个 shape 数组。
    SolasDType dtype;
    std::vector<float> data;
};//头文件对外隐藏结构体，.cpp 内部必须先完整定义结构体，才能访问字段。如果这段放到namespace后面，编译报错

namespace {//只给当前 .cpp 文件内部使用的工具和状态。

thread_local std::string last_error;

void clear_error() {
    last_error.clear();
}

SolasStatus set_error(const char *message) {
    last_error = message == nullptr ? "" : message;
    return SOLAS_STATUS_ERROR;
}

long long dtype_size(SolasDType dtype) {
    switch (dtype) {
        case SOLAS_DTYPE_FLOAT32:
            return 4;
    }

    return 0;
}

SolasStatus validate_binary_float32_op(
    const char *op_name,
    const SolasTensor *lhs,
    const SolasTensor *rhs,
    const SolasTensor *out
) {
    if (lhs == nullptr) {
        last_error = std::string(op_name) + " received null lhs tensor";
        return SOLAS_STATUS_ERROR;
    }

    if (rhs == nullptr) {
        last_error = std::string(op_name) + " received null rhs tensor";
        return SOLAS_STATUS_ERROR;
    }

    if (out == nullptr) {
        last_error = std::string(op_name) + " received null output tensor";
        return SOLAS_STATUS_ERROR;
    }

    if (lhs->dtype != SOLAS_DTYPE_FLOAT32 ||
        rhs->dtype != SOLAS_DTYPE_FLOAT32 ||
        out->dtype != SOLAS_DTYPE_FLOAT32) {
        last_error = std::string(op_name) + " only supports float32 tensors";
        return SOLAS_STATUS_ERROR;
    }

    if (lhs->shape != rhs->shape || lhs->shape != out->shape) {
        last_error = std::string(op_name) + " requires matching shapes";
        return SOLAS_STATUS_ERROR;
    }

    if (lhs->data.size() != rhs->data.size() ||
        lhs->data.size() != out->data.size()) {
        last_error = std::string(op_name) + " requires matching data sizes";
        return SOLAS_STATUS_ERROR;
    }

    return SOLAS_STATUS_OK;
}

float add_float32(float lhs, float rhs) {
    return lhs + rhs;
}

float mul_float32(float lhs, float rhs) {
    return lhs * rhs;
}

float sub_float32(float lhs, float rhs) {
    return lhs - rhs;
}

float div_float32(float lhs, float rhs) {
    return lhs / rhs;
}

SolasStatus binary_float32_op(
    const char *op_name,
    const SolasTensor *lhs,
    const SolasTensor *rhs,
    SolasTensor *out,
    float (*op)(float, float)//表示 op 是一个函数指针，指向这种函数：float something(float, float)
) {
    clear_error();

    SolasStatus status = validate_binary_float32_op(op_name, lhs, rhs, out);
    if (status != SOLAS_STATUS_OK) {
        return status;
    }

    for (size_t i = 0; i < out->data.size(); ++i) {
        out->data[i] = op(lhs->data[i], rhs->data[i]);
    }

    return SOLAS_STATUS_OK;
}

SolasStatus validate_unary_float32_op(
    const char *op_name,
    const SolasTensor *input,
    const SolasTensor *out
) {
    if (input == nullptr) {
        last_error = std::string(op_name) + " received null input tensor";
        return SOLAS_STATUS_ERROR;
    }

    if (out == nullptr) {
        last_error = std::string(op_name) + " received null output tensor";
        return SOLAS_STATUS_ERROR;
    }

    if (input->dtype != SOLAS_DTYPE_FLOAT32 ||
        out->dtype != SOLAS_DTYPE_FLOAT32) {
        last_error = std::string(op_name) + " only supports float32 tensors";
        return SOLAS_STATUS_ERROR;
    }

    if (input->shape != out->shape) {
        last_error = std::string(op_name) + " requires matching shapes";
        return SOLAS_STATUS_ERROR;
    }

    if (input->data.size() != out->data.size()) {
        last_error = std::string(op_name) + " requires matching data sizes";
        return SOLAS_STATUS_ERROR;
    }

    return SOLAS_STATUS_OK;
}

float neg_float32(float value) {
    return -value;
}

SolasStatus unary_float32_op(
    const char *op_name,
    const SolasTensor *input,
    SolasTensor *out,
    float (*op)(float)
) {
    clear_error();

    SolasStatus status = validate_unary_float32_op(op_name, input, out);
    if (status != SOLAS_STATUS_OK) {
        return status;
    }

    for (size_t i = 0; i < out->data.size(); ++i) {
        out->data[i] = op(input->data[i]);
    }

    return SOLAS_STATUS_OK;
}

SolasStatus scalar_float32_op(
    const char *op_name,
    const SolasTensor *input,
    float scalar,
    SolasTensor *out,
    float (*op)(float, float)
) {
    clear_error();

    SolasStatus status = validate_unary_float32_op(op_name, input, out);
    if (status != SOLAS_STATUS_OK) {
        return status;
    }

    for (size_t i = 0; i < out->data.size(); ++i) {
        out->data[i] = op(input->data[i], scalar);
    }

    return SOLAS_STATUS_OK;
}

}  // namespace



const char *solas_version(void) {
    return "0.0.0";
}

const char *solas_last_error(void) {
    return last_error.c_str();
}

SolasStatus solas_tensor_create(const long long *shape, int rank, SolasDType dtype,  SolasTensor **out) {
    clear_error();

    if (out == nullptr) {
        return set_error("solas_tensor_create received null output pointer");
    }

    if (rank < 0) {
        return set_error("solas_tensor_create received negative rank");
    }

    if (rank > 0 && shape == nullptr) {
        return set_error("solas_tensor_create received null shape for non-scalar tensor");
    }

    if (dtype != SOLAS_DTYPE_FLOAT32) {
        return set_error("solas_tensor_create received unsupported dtype");
    }

    for (int i = 0; i < rank; ++i) {
        if (shape[i] < 0) {
            return set_error("solas_tensor_create received negative shape dimension");
        }
    }

    SolasTensor *tensor = new (std::nothrow) SolasTensor{};

    if (tensor == nullptr) {
        *out = nullptr;
        return set_error("failed to allocate SolasTensor");
    }

    tensor->dtype = dtype;

    if (rank > 0) {
        tensor->shape.assign(shape, shape + rank);//?
    }
    long long numel = 1;
    for (long long dim : tensor->shape) {
        numel *= dim;
    }
     tensor->data.assign(static_cast<size_t>(numel), 0.0f);

    *out = tensor;
    return SOLAS_STATUS_OK;
}

void solas_tensor_destroy(SolasTensor *tensor) {
    delete tensor;
}

SolasStatus solas_tensor_rank(const SolasTensor *tensor, int *out) {
    clear_error();

    if (tensor == nullptr) {
        return set_error("solas_tensor_rank received null tensor");
    }

    if (out == nullptr) {
        return set_error("solas_tensor_rank received null output pointer");
    }

    *out = static_cast<int>(tensor->shape.size());
    return SOLAS_STATUS_OK;
}

SolasStatus solas_tensor_dim(const SolasTensor *tensor, int index, long long *out) {
    clear_error();

    if (tensor == nullptr) {
        return set_error("solas_tensor_dim received null tensor");
    }

    if (out == nullptr) {
        return set_error("solas_tensor_dim received null output pointer");
    }

    if (index < 0 || index >= static_cast<int>(tensor->shape.size())) {
        return set_error("solas_tensor_dim received out-of-range index");
    }

    *out = tensor->shape[static_cast<size_t>(index)];
    return SOLAS_STATUS_OK;
}

SolasStatus solas_tensor_numel(const SolasTensor *tensor, long long *out) {
    clear_error();

    if (tensor == nullptr) {
        return set_error("solas_tensor_numel received null tensor");
    }

    if (out == nullptr) {
        return set_error("solas_tensor_numel received null output pointer");
    }

    long long numel = 1; //标量 Tensor 的 shape 是空的：空 shape 的元素数量是 1，所以初始值是 1。
    for (long long dim : tensor->shape) {
        numel *= dim;
    }

    *out = numel;
    return SOLAS_STATUS_OK;
}

//查询 dtype
SolasStatus solas_tensor_dtype(const SolasTensor *tensor, SolasDType *out) {
    clear_error();

    if (tensor == nullptr) {
        return set_error("solas_tensor_dtype received null tensor");
    }

    if (out == nullptr) {
        return set_error("solas_tensor_dtype received null output pointer");
    }

    *out = tensor->dtype;
    return SOLAS_STATUS_OK;
}


SolasStatus solas_tensor_nbytes(const SolasTensor *tensor, long long *out) {
    clear_error();

    if (tensor == nullptr) {
        return set_error("solas_tensor_nbytes received null tensor");
    }

    if (out == nullptr) {
        return set_error("solas_tensor_nbytes received null output pointer");
    }

    long long numel = 1;
    for (long long dim : tensor->shape) {
        numel *= dim;
    }

    *out = numel * dtype_size(tensor->dtype);
    return SOLAS_STATUS_OK;
}

SolasStatus solas_tensor_data_size(const SolasTensor *tensor, long long *out) {
    clear_error();

    if (tensor == nullptr) {
        return set_error("solas_tensor_data_size received null tensor");
    }

    if (out == nullptr) {
        return set_error("solas_tensor_data_size received null output pointer");
    }

    *out = static_cast<long long>(tensor->data.size());
    return SOLAS_STATUS_OK;
}

SolasStatus solas_tensor_fill_float32(SolasTensor *tensor, float value) {
    clear_error();

    if (tensor == nullptr) {
        return set_error("solas_tensor_fill_float32 received null tensor");
    }

    if (tensor->dtype != SOLAS_DTYPE_FLOAT32) {
        return set_error("solas_tensor_fill_float32 only supports float32 tensors");
    }

    for (float &item : tensor->data) {
        item = value;
    }

    return SOLAS_STATUS_OK;
}

SolasStatus solas_tensor_get_float32(const SolasTensor *tensor, long long index, float *out) {
    clear_error();

    if (tensor == nullptr) {
        return set_error("solas_tensor_get_float32 received null tensor");
    }

    if (out == nullptr) {
        return set_error("solas_tensor_get_float32 received null output pointer");
    }

    if (tensor->dtype != SOLAS_DTYPE_FLOAT32) {
        return set_error("solas_tensor_get_float32 only supports float32 tensors");
    }

    if (index < 0 || index >= static_cast<long long>(tensor->data.size())) {
        return set_error("solas_tensor_get_float32 received out-of-range index");
    }

    *out = tensor->data[static_cast<size_t>(index)];
    return SOLAS_STATUS_OK;
}


SolasStatus solas_tensor_set_float32(SolasTensor *tensor, long long index, float value) {
    clear_error();

    if (tensor == nullptr) {
        return set_error("solas_tensor_set_float32 received null tensor");
    }

    if (tensor->dtype != SOLAS_DTYPE_FLOAT32) {
        return set_error("solas_tensor_set_float32 only supports float32 tensors");
    }

    if (index < 0 || index >= static_cast<long long>(tensor->data.size())) {
        return set_error("solas_tensor_set_float32 received out-of-range index");
    }

    tensor->data[static_cast<size_t>(index)] = value;
    return SOLAS_STATUS_OK;
}

SolasStatus solas_tensor_add_float32(
    const SolasTensor *lhs,
    const SolasTensor *rhs,
    SolasTensor *out
) {
    return binary_float32_op("solas_tensor_add_float32", lhs, rhs, out, add_float32);
}

SolasStatus solas_tensor_mul_float32(
    const SolasTensor *lhs,
    const SolasTensor *rhs,
    SolasTensor *out
) {
    return binary_float32_op("solas_tensor_mul_float32", lhs, rhs, out, mul_float32);
}

SolasStatus solas_tensor_sub_float32(
    const SolasTensor *lhs,
    const SolasTensor *rhs,
    SolasTensor *out
) {
    return binary_float32_op("solas_tensor_sub_float32", lhs, rhs, out, sub_float32);
}

SolasStatus solas_tensor_div_float32(
    const SolasTensor *lhs,
    const SolasTensor *rhs,
    SolasTensor *out
) {
    return binary_float32_op("solas_tensor_div_float32", lhs, rhs, out, div_float32);
}

SolasStatus solas_tensor_neg_float32(
    const SolasTensor *input,
    SolasTensor *out
) {
    return unary_float32_op("solas_tensor_neg_float32", input, out, neg_float32);
}

SolasStatus solas_tensor_add_scalar_float32(
    const SolasTensor *input,
    float scalar,
    SolasTensor *out
) {
    return scalar_float32_op(
        "solas_tensor_add_scalar_float32",
        input,
        scalar,
        out,
        add_float32
    );
}