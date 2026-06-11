from __future__ import annotations

from solas.libsolas import (
    solas_tensor_create,
    solas_tensor_destroy,
    solas_tensor_dim,
    solas_tensor_rank,
)
class Tensor:
    def __init__(self) -> None:
        self._handle = solas_tensor_create()

    @property
    def rank(self) -> int:
        return solas_tensor_rank(self._handle)
    
    @property
    def shape(self) -> tuple[int, ...]:
        rank = self.rank
        return tuple(solas_tensor_dim(self._handle, index) for index in range(rank))

    def close(self) -> None:
        if self._handle is not None:
            solas_tensor_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Tensor(handle={self._handle})"