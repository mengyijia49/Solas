
from __future__ import annotations

import ctypes
import os
from pathlib import Path


def _find_library() -> Path:
    configured_path = os.environ.get("SOLAS_LIBRARY_PATH")
    if configured_path:
        path = Path(configured_path).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"SOLAS_LIBRARY_PATH does not exist: {path}")

    project_root = Path(__file__).resolve().parents[3]

    candidates = [
        project_root / "build" / "linux" / "x86_64" / "release" / "libsolas.so",
        project_root / "build" / "linux" / "x86_64" / "debug" / "libsolas.so",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError("libsolas.so not found. Run `xmake` first.")


_lib = ctypes.CDLL(str(_find_library()))

SOLAS_DTYPE_FLOAT32 = 0

_lib.solas_version.restype = ctypes.c_char_p
_lib.solas_last_error.restype = ctypes.c_char_p#告诉ctypes:solas_last_error 这个 C 函数返回 char *

_lib.solas_tensor_create.argtypes = [ctypes.POINTER(ctypes.c_longlong),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_void_p)]
_lib.solas_tensor_create.restype = ctypes.c_int

_lib.solas_tensor_destroy.argtypes = [ctypes.c_void_p]
_lib.solas_tensor_destroy.restype = None

_lib.solas_tensor_rank.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_lib.solas_tensor_rank.restype = ctypes.c_int

_lib.solas_tensor_dim.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
_lib.solas_tensor_dim.restype = ctypes.c_int

_lib.solas_tensor_numel.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong)]
_lib.solas_tensor_numel.restype = ctypes.c_int

_lib.solas_tensor_dtype.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_lib.solas_tensor_dtype.restype = ctypes.c_int

_lib.solas_tensor_nbytes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong)]
_lib.solas_tensor_nbytes.restype = ctypes.c_int

_lib.solas_tensor_data_size.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_longlong)]
_lib.solas_tensor_data_size.restype = ctypes.c_int


_lib.solas_tensor_fill_float32.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.solas_tensor_fill_float32.restype = ctypes.c_int

_lib.solas_tensor_get_float32.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.POINTER(ctypes.c_float),]
_lib.solas_tensor_get_float32.restype = ctypes.c_int

_lib.solas_tensor_set_float32.argtypes = [ctypes.c_void_p,ctypes.c_longlong,ctypes.c_float,]
_lib.solas_tensor_set_float32.restype = ctypes.c_int

_lib.solas_tensor_add_float32.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,]
_lib.solas_tensor_add_float32.restype = ctypes.c_int

def solas_version() -> str:
    return _lib.solas_version().decode("utf-8")

def solas_last_error() -> str:
    return _lib.solas_last_error().decode("utf-8")


def solas_tensor_create(shape: tuple[int, ...] = (), dtype: int = SOLAS_DTYPE_FLOAT32) -> int:# 默认 shape 是空 tuple，也就是标量。
    tensor = ctypes.c_void_p()                              # 1. 创建一个空的 c_void_p
    rank = len(shape)                                       # rank 来自 Python tuple 的长度。

    if rank == 0:
        shape_ptr = None                                    # 对于标量 Tensor，传给 C 的 shape 指针是 NULL。
    else:
        shape_array = (ctypes.c_longlong * rank)(*shape)    # 把 Python tuple 转成 C 数组。例如(2, 3)变成 C 里的：long long shape[2] = {2, 3};
        shape_ptr = shape_array

    status = _lib.solas_tensor_create(shape_ptr, ctypes.c_int(rank),ctypes.c_int(dtype), ctypes.byref(tensor))
    if status != 0:
        raise RuntimeError(solas_last_error())
    return tensor.value

def solas_tensor_destroy(handle: int | None) -> None:
    _lib.solas_tensor_destroy(ctypes.c_void_p(handle))

def solas_tensor_rank(handle: int | None) -> int:
    rank = ctypes.c_int()
    status = _lib.solas_tensor_rank(ctypes.c_void_p(handle), ctypes.byref(rank))
    if status != 0:
        raise RuntimeError(solas_last_error())
    return rank.value

def solas_tensor_dim(handle: int | None, index: int) -> int:
    dim = ctypes.c_longlong()
    status = _lib.solas_tensor_dim(
        ctypes.c_void_p(handle),
        ctypes.c_int(index),
        ctypes.byref(dim),
    )
    if status != 0:
        raise RuntimeError(solas_last_error())
    return dim.value

def solas_tensor_numel(handle: int | None) -> int:
    numel = ctypes.c_longlong()
    status = _lib.solas_tensor_numel(ctypes.c_void_p(handle), ctypes.byref(numel))
    if status != 0:
        raise RuntimeError(solas_last_error())
    return numel.value


def solas_tensor_dtype(handle: int | None) -> int:
    dtype = ctypes.c_int()
    status = _lib.solas_tensor_dtype(ctypes.c_void_p(handle), ctypes.byref(dtype))
    if status != 0:
        raise RuntimeError(solas_last_error())
    return dtype.value

def solas_tensor_nbytes(handle: int | None) -> int:
    nbytes = ctypes.c_longlong()
    status = _lib.solas_tensor_nbytes(ctypes.c_void_p(handle), ctypes.byref(nbytes))
    if status != 0:
        raise RuntimeError(solas_last_error())
    return nbytes.value

def solas_tensor_data_size(handle: int | None) -> int:
    size = ctypes.c_longlong()
    status = _lib.solas_tensor_data_size(ctypes.c_void_p(handle), ctypes.byref(size))
    if status != 0:
        raise RuntimeError(solas_last_error())
    return size.value

def solas_tensor_fill_float32(handle: int | None, value: float) -> None:
    status = _lib.solas_tensor_fill_float32(ctypes.c_void_p(handle), ctypes.c_float(value))
    if status != 0:
        raise RuntimeError(solas_last_error())

def solas_tensor_get_float32(handle: int | None, index: int) -> float:
    value = ctypes.c_float()
    status = _lib.solas_tensor_get_float32(
        ctypes.c_void_p(handle),
        ctypes.c_longlong(index),
        ctypes.byref(value),
    )
    if status != 0:
        raise RuntimeError(solas_last_error())
    return value.value

def solas_tensor_set_float32(handle: int | None, index: int, value: float) -> None:
    status = _lib.solas_tensor_set_float32(
        ctypes.c_void_p(handle),
        ctypes.c_longlong(index),
        ctypes.c_float(value),
    )
    if status != 0:
        raise RuntimeError(solas_last_error())
    

def solas_tensor_add_float32(lhs: int | None, rhs: int | None, out: int | None) -> None:
    status = _lib.solas_tensor_add_float32(
        ctypes.c_void_p(lhs),
        ctypes.c_void_p(rhs),
        ctypes.c_void_p(out),
    )
    if status != 0:
        raise RuntimeError(solas_last_error())