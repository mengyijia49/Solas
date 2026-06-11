# Solas Architecture

Solas 采用 Python 前端 + C/C++ 后端的分层结构。

Python 层负责用户接口、测试入口和模型调用流程。
C/C++ 层负责张量、内存管理、算子和设备运行时。
Python 和 C/C++ 之间通过 C API 连接。


## Directory Layout

```text
include/solas/        C API 头文件
src/solas/            C API 实现
src/core/             核心运行时
src/tensor/           Tensor、Shape、Storage 等基础结构
src/ops/              算子实现
src/device/cpu/       CPU 后端
python/solas/         Python 前端
python/solas/libsolas ctypes 封装层
test/                 测试
docs/                 文档
```

## First Backend

Solas 的第一个后端是 CPU 后端。

初期目标不是性能，而是把推理框架的基本路径跑通：

1. Python 创建 Tensor。
2. Python 调用 C API。
3. C/C++ 后端创建底层 Tensor。
4. C/C++ 后端执行基础算子。
5. Python 读取结果并测试。

## Long Term Direction

后续 Solas 会逐步加入：

- 手写 tokenizer。
- Tensor 存储和 shape 系统。
- 基础算子。
- 矩阵乘法。
- softmax 和 layer normalization。
- attention。
- transformer block。
- 模型权重加载。
- 文本生成流程。