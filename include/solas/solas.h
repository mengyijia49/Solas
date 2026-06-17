#ifndef SOLAS_SOLAS_H
#define SOLAS_SOLAS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SolasStatus {
    SOLAS_STATUS_OK = 0,
    SOLAS_STATUS_ERROR = 1,
} SolasStatus;

typedef enum SolasDType {
    SOLAS_DTYPE_FLOAT32 = 0,
} SolasDType;

typedef struct SolasTensor SolasTensor;//外部世界知道有一个类型叫 SolasTensor，但不知道它里面长什么样。

const char *solas_version(void);//返回当前 Solas 版本。
const char *solas_last_error(void);//返回最近一次 C/C++ 后端错误信息。


SolasStatus solas_tensor_create(const long long *shape, int rank, SolasDType dtype, SolasTensor **out);//创建一个Tensor
void solas_tensor_destroy(SolasTensor *tensor);//销毁tensor

SolasStatus solas_tensor_rank(const SolasTensor *tensor, int *out);//获取 Tensor 的维度数量。scalar: rank = 0,vector: rank = 1,matrix: rank = 2
SolasStatus solas_tensor_dim(const SolasTensor *tensor, int index, long long *out);//获取第 index 维的大小。
SolasStatus solas_tensor_numel(const SolasTensor *tensor, long long *out);
SolasStatus solas_tensor_dtype(const SolasTensor *tensor, SolasDType *out);
SolasStatus solas_tensor_nbytes(const SolasTensor *tensor, long long *out);
SolasStatus solas_tensor_data_size(const SolasTensor *tensor, long long *out);
SolasStatus solas_tensor_fill_float32(SolasTensor *tensor, float value);
SolasStatus solas_tensor_get_float32(const SolasTensor *tensor, long long index, float *out);//读取第index个元素写到out。这里是一维扁平index，暂时不做多维索引
SolasStatus solas_tensor_set_float32(SolasTensor *tensor, long long index, float value);//把 float32 Tensor 的第 index 个元素设置为 value，扁平索引。
SolasStatus solas_tensor_add_float32(const SolasTensor *lhs,const SolasTensor *rhs,SolasTensor *out);

#ifdef __cplusplus
}
#endif

#endif