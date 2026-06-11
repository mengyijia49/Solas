#ifndef SOLAS_SOLAS_H
#define SOLAS_SOLAS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum SolasStatus {
    SOLAS_STATUS_OK = 0,
    SOLAS_STATUS_ERROR = 1,
} SolasStatus;

typedef struct SolasTensor SolasTensor;//外部世界知道有一个类型叫 SolasTensor，但不知道它里面长什么样。

const char *solas_version(void);//返回当前 Solas 版本。
const char *solas_last_error(void);//返回最近一次 C/C++ 后端错误信息。


SolasStatus solas_tensor_create(SolasTensor **out);//创建一个Tensor
void solas_tensor_destroy(SolasTensor *tensor);//销毁tensor

SolasStatus solas_tensor_rank(const SolasTensor *tensor, int *out);//获取 Tensor 的维度数量。scalar: rank = 0,vector: rank = 1,matrix: rank = 2
SolasStatus solas_tensor_dim(const SolasTensor *tensor, int index, long long *out);//获取第 index 维的大小。

#ifdef __cplusplus
}
#endif

#endif