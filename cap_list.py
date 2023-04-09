import sys

from typing import Iterator, Any


class CapList(list):
    def __init__(self, cap: int = 0, data: Iterator[Any]=None):
        self._size = 0
        self._cap = cap
        self._data = None
        if self._cap > 0 and data is None:
            self._data = [None] * self._cap
        if data is not None:
            self._data = list(data)
            self._cap = len(self._data)
        if self._data is None:
            self._data = []

    def __len__(self) -> int:
        return self._size
    
    def __iter__(self) -> Iterator[Any]:
        for i in range(0, self._size):
            yield self._data[i]

    def __getitem__(self, idx: int) -> Any:
        return self._data[idx]
    
    def __setitem__(self, idx: int, __object: Any) -> None:
        self._data[idx] = __object

    def append(self, __object: Any) -> None:
        if self._size < self._cap:
            self._data[self._size] = __object
            self._size += 1
            return

        super().append(__object)
        self._size += 1
        self._cap = len(self._data)

    def pop(self, __index: int = -1) -> Any:
        if super().pop(__index) is not None:
            self._cap -= 1
            self._size -= 1

    def remove(self, __value: Any) -> None:
        super().remove(__value)
        if self._cap != len(self._data):
            self._size -= 1
            self._cap -= 1 
