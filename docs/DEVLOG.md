# Solas 开发日志

## 2026-06-11

- 创建了初始项目结构。
- 添加了第一个 C API 头文件：`include/solas/solas.h`。
- 添加了第一个 C++ 实现：`src/solas/solas.cpp`。
- 添加了 `xmake.lua`，用于构建 `libsolas`。
- 在 `python/solas/libsolas` 下添加了 Python `ctypes` 绑定。
- 添加了第一个端到端版本测试。
- 添加了 `solas_last_error` C API。
- 在 Python 中暴露了 `solas.last_error()` 并进行了测试。
- 添加了不透明 `SolasTensor` 句柄。
- 添加了张量创建和销毁 C API。
- 添加了 Python `Tensor` 包装器。
- 添加了张量生命周期测试。
- 添加了张量秩和维度 C API。
- 在 Python 中暴露了 `Tensor.rank` 和 `Tensor.shape`。
- 添加了标量张量形状测试。
