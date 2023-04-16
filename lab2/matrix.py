

from io import IOBase
from typing import List, Set, Union, Tuple, Dict

from multiprocess.shared_memory import SharedMemory

import numpy as np

class Matrix(object):
    def __init__(self, mtx: List[List[int]], mounted: List[int]=[]):
        self._n: int = len(mtx)
        
        self._dtype = np.int32

        self._mtx: List[List[int]] = np.ndarray((self._n, self._n), dtype=self._dtype)
        self._mounted = np.array(mounted, dtype=self._dtype)
        self._mounted.sort()
        
        self._cross_list: List[Set[int]] = [set() for _ in range(self._n)]
        # self._degrees: List = [None] * self._n 

        for i, row in enumerate(mtx):
            for j, x in enumerate(row):
                self._mtx[i, j] = mtx[i][j]
                if x > 0:
                    self._cross_list[i].add(j)
                    self._cross_list[j].add(i)
            # self._degrees[i] = np.sum(self._cross_list[i], where=lambda x: (self._mounted.searchsorted(x) is None))

    @property
    def n(self) -> int: return self._n

    @property
    def mtx(self) -> np.ndarray: return self._mtx

    @property
    def cross_list(self) -> List[Set[int]]: return self._cross_list

    # @property
    # def degrees(self) -> List[int]: return self._degrees

    @property
    def dtype(self): return self._dtype

    @property
    def mounted(self) -> np.ndarray: return self._mounted

    def crosses(self, x: int) -> Set[int]: return self._cross_list[x]


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

