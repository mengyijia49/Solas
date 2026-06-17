# Solas Roadmap

Solas will grow in small, testable steps.

## Phase 0: Project Foundation

- [x] Create repository structure.
- [x] Add README, license, and project metadata.
- [x] Add the first C API header.
- [x] Add the first C++ implementation file.
- [x] Build `libsolas` with xmake.
- [x] Add Python `ctypes` binding.
- [x] Add the first end-to-end test.


## Phase 1: Runtime Skeleton

- [x] Define `SolasStatus`.
- [x] Add error message API.
- [x] Add opaque `SolasTensor` handle.
- [x] Add tensor create and destroy APIs.
- [x] Add tensor shape APIs.
- [x] Add Python Tensor wrapper.
- [x] Add tensor lifecycle tests.


## Phase 2: Tensor Core

- [x] Store tensor dtype.
- [x] Add tensor byte-size API.
- [x] Store tensor shape.
- [x] Store tensor data on CPU.
- [x] Implement scalar tensor.
- [x] Implement 1D tensor.
- [x] Implement 2D tensor.
- [x] Add tensor data read API.
- [x] Add tensor data write API.
- [x] Add tensor fill API.
- [x] Add Python `Tensor.to_list()`.

## Phase 3: Operators

- [x] Add elementwise add.
- [x] Add elementwise multiply.
- [x] Add elementwise subtract.
- [ ] Add matrix multiplication.
- [ ] Add softmax.
- [ ] Add layer normalization.