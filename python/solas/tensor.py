from __future__ import annotations

from solas.libsolas import (
    SOLAS_DTYPE_FLOAT32,
    solas_tensor_create,
    solas_tensor_data_size,
    solas_tensor_destroy,
    solas_tensor_dim,
    solas_tensor_dtype,
    solas_tensor_nbytes,
    solas_tensor_numel,
    solas_tensor_rank,
    solas_tensor_fill_float32,
    solas_tensor_get_float32,
    solas_tensor_set_float32,
    solas_tensor_add_float32,
)

_DTYPE_TO_ID = {"float32": SOLAS_DTYPE_FLOAT32,}

_ID_TO_DTYPE = {SOLAS_DTYPE_FLOAT32: "float32",}
class Tensor:
    def __init__(self, shape: tuple[int, ...] = (), dtype: str = "float32") -> None: # 暂时只支持 tuple
        self._handle: int | None = None                      #类型是int 或 None
        if dtype not in _DTYPE_TO_ID:
            raise ValueError(f"unsupported dtype: {dtype}")
        self._handle = solas_tensor_create(shape, _DTYPE_TO_ID[dtype])

    @property
    def rank(self) -> int:
        return solas_tensor_rank(self._handle)
    
    @property
    def shape(self) -> tuple[int, ...]:
        rank = self.rank
        return tuple(solas_tensor_dim(self._handle, index) for index in range(rank))
    
    @property
    def numel(self) -> int:
        return solas_tensor_numel(self._handle)
    
    @property
    def nbytes(self) -> int:
        return solas_tensor_nbytes(self._handle)
    
    @property
    def data_size(self) -> int:
        return solas_tensor_data_size(self._handle)
    
    @property
    def dtype(self) -> str:
        dtype_id = solas_tensor_dtype(self._handle)
        return _ID_TO_DTYPE[dtype_id]
    
    def fill(self, value: float) -> None:
        solas_tensor_fill_float32(self._handle, value)


    def get(self, index: int) -> float:
        return solas_tensor_get_float32(self._handle, index)
    
    def set(self, index: int, value: float) -> None:
        solas_tensor_set_float32(self._handle, index, value)

    def add(self, other: "Tensor") -> "Tensor":
        if self.shape != other.shape:
            raise ValueError("Tensor.add requires matching shapes")

        if self.dtype != other.dtype:
            raise ValueError("Tensor.add requires matching dtypes")

        out = Tensor(self.shape, dtype=self.dtype)
        solas_tensor_add_float32(self._handle, other._handle, out._handle)
        return out

    def __add__(self, other: "Tensor") -> "Tensor":
        return self.add(other)

    def to_list(self) -> object:
        if self.rank == 0:
            return self.get(0)#标量直接读取第0个元素

        return self._to_nested_list(0, 0)[0]
    
    def _to_nested_list(self, dim: int, offset: int) -> tuple[object, int]:
        if dim == self.rank:
            return self.get(offset), offset + 1

        items = []
        next_offset = offset
        for _ in range(self.shape[dim]):
            item, next_offset = self._to_nested_list(dim + 1, next_offset)
            items.append(item)

        return items, next_offset

    def close(self) -> None:
        if self._handle is not None:
            solas_tensor_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"Tensor(shape={self.shape}, dtype={self.dtype!r}, "f"numel={self.numel}, nbytes={self.nbytes}, "f"data_size={self.data_size}, handle={self._handle})")

