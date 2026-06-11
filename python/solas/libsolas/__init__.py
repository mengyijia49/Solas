
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
_lib.solas_version.restype = ctypes.c_char_p
_lib.solas_last_error.restype = ctypes.c_char_p#告诉ctypes:solas_last_error 这个 C 函数返回 char *

_lib.solas_tensor_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_lib.solas_tensor_create.restype = ctypes.c_int

_lib.solas_tensor_destroy.argtypes = [ctypes.c_void_p]
_lib.solas_tensor_destroy.restype = None

_lib.solas_tensor_rank.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
_lib.solas_tensor_rank.restype = ctypes.c_int

_lib.solas_tensor_dim.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_longlong)]
_lib.solas_tensor_dim.restype = ctypes.c_int

def solas_version() -> str:
    return _lib.solas_version().decode("utf-8")

def solas_last_error() -> str:
    return _lib.solas_last_error().decode("utf-8")

def solas_tensor_create() -> int:
    tensor = ctypes.c_void_p()                              # 1. 创建一个空的 c_void_p
    status = _lib.solas_tensor_create(ctypes.byref(tensor)) # 2. 把它的地址传给 C API，3. C++ 把 Tensor 指针写进去
    if status != 0:                                         # 4. 如果失败，抛 RuntimeError
        raise RuntimeError(solas_last_error())              #
    return tensor.value                                     # 5. 如果成功，返回指针值

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

