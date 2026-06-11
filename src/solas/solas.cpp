#include "solas/solas.h"

#include <new>
#include <string>
#include <vector>

namespace {//只给当前 .cpp 文件内部使用的工具和状态。

thread_local std::string last_error;

void clear_error() {
    last_error.clear();
}

SolasStatus set_error(const char *message) {
    last_error = message == nullptr ? "" : message;
    return SOLAS_STATUS_ERROR;
}

}  // namespace

struct SolasTensor {
    std::vector<long long> shape;//表示每个 Tensor 内部保存一个 shape 数组。
};

const char *solas_version(void) {
    return "0.0.0";
}

const char *solas_last_error(void) {
    return last_error.c_str();
}

SolasStatus solas_tensor_create(const long long *shape, int rank, SolasTensor **out) {
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

    if (rank > 0) {
        tensor->shape.assign(shape, shape + rank);//?
    }

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