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

- [ ] Store tensor dtype.
- [ ] Store tensor shape.
- [ ] Store tensor data on CPU.
- [ ] Implement scalar tensor.
- [ ] Implement 1D tensor.
- [ ] Implement 2D tensor.
- [ ] Add tensor data read/write APIs.


## Phase 3: Operators

- [ ] Add elementwise add.
- [ ] Add elementwise multiply.
- [ ] Add matrix multiplication.
- [ ] Add softmax.
- [ ] Add layer normalization.