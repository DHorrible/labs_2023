

from io import IOBase
from typing import List, Set, Union, Tuple, Dict

from multiprocess.shared_memory import SharedMemory

import numpy as np

class Matrix(object):
    def __init__(self, mtx: List[List[int]]):
        self._n: int = len(mtx)
        
        self._mtx: List[List[int]] = np.ndarray((self._n, self._n), dtype=np.int32)
        self._cross_list: List[Set[int]] = [set() for _ in range(self._n)]
        self._degrees: List = [None] * self._n 

        for i, row in enumerate(mtx):
            for j, x in enumerate(row):
                self._mtx[i, j] = mtx[i][j]
                if x > 0:
                    self._cross_list[i].add(j)
                    self._cross_list[j].add(i)
            self._degrees[i] = sum(row)

        # self._swap_index: Dict[int, int] = {x: x for x in range(self._n)} 

    @property
    def n(self) -> int: return self._n

    @property
    def mtx(self) -> np.ndarray: return self._mtx

    @property
    def cross_list(self) -> List[Set[int]]: return self._cross_list
 
    @property
    def degrees(self) -> List[int]: return self._degrees

    def set_shared_mem(self, shm: SharedMemory) -> None:
        mtx = np.ndarray((self._n, self._n), dtype=np.int32, buffer=shm.buf)
        mtx[:] = self._mtx[:]
        self._mtx = mtx

    def crosses(self, x: int) -> Set[int]: return self._cross_list[x]

    # def swap_cols(self, x: int, y: int) -> None:
    #     for i in range(self.n):
    #         self._mtx[i][x], self._mtx[i][y] = self._mtx[i][y], self._mtx[i][x]

    # def swap_rows(self, x: int, y: int) -> None:
    #     self._mtx[x], self._mtx[y] = self._mtx[y], self._mtx[x]

    def swap_nodes(self, x: int, y: int) -> None:
        self._swap_index[x], self._swap_index[y] = self._swap_index[y], self._swap_index[x]

        # self.swap_cols(x, y)
        # self.swap_rows(x, y)

    def at_image(self, i: int, j: int) -> int:
        return self._mtx[self._swap_index[i]][self._swap_index[j]]

class StreamMatrix(Matrix):
    def __init__(self, stream: IOBase):
        mtx = []
        while stream.readable():
            line = stream.readline()
            if line is None or line == '':
                break
            row = list(map(int, line.replace(' ', '').split(',')))
            mtx.append(row)
    
        super().__init__(mtx=mtx)

