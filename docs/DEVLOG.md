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
- 通过 C API 增加了 Tensor 形状创建功能。
- 增加了 Python Tensor(shape) 支持。
- 增加了标量 Tensor 和有形状 Tensor 的测试。
- 增加了负数形状校验和错误传播测试。


## 2026-06-14
- Added `solas_tensor_numel` C API.
- Exposed `Tensor.numel` in Python.
- Added tests for scalar, shaped, and empty tensors.

## 2026-06-15
- Added `SolasDType` with initial `float32` support.
- Added dtype storage to `SolasTensor`.
- Exposed `Tensor.dtype` in Python.
- Added tests for default dtype and unsupported dtype validation.
- Added `solas_tensor_nbytes` C API.
- Exposed `Tensor.nbytes` in Python.
- Added byte-size tests for scalar, shaped, and empty tensors.
- Added CPU float32 storage to `SolasTensor`.
- Added `solas_tensor_data_size` C API.
- Exposed `Tensor.data_size` in Python.
- Added tests to verify storage size matches `numel`.
- Added `solas_tensor_fill_float32` C API.
- Added `solas_tensor_get_float32` C API.
- Exposed `Tensor.fill()` and `Tensor.get()` in Python.
- Added tests for filled values and out-of-range reads.
- Added `solas_tensor_set_float32` C API.
- Exposed `Tensor.set()` in Python.
- Added tests for single-element writes and out-of-range writes.
- Added Python `Tensor.to_list()` for scalar, 1D, and 2D tensors.
- Added tests for scalar, vector, and matrix list conversion.
- Added `scripts/test.sh` to build the backend and run the Python test suite.