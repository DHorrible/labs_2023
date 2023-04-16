import numpy as np

from typing import List, Tuple

from matrix import Matrix

class RouteMtxBase(object):
    def __init__(self, n: int, dtype=None) -> None:
        self._n = n
        self._mtx = np.ndarray((n, n), dtype=dtype)

    @property
    def n(self) -> int: return self._n

    @property
    def mtx(self) -> np.ndarray: return self._mtx

class SqauredRouteMtx(RouteMtxBase):
    def __init__(self, n: int, dtype=None) -> None:
        super().__init__(n, dtype)
        self._setup_mtx()

    def _setup_mtx(self) -> None:
        # TODO :
        height, weight = self._get_place_shape()
        for i in range(self.n):
            i_row, i_col = i // height, i % weight
            for j in range(self.n):
                if i == j:
                    self._mtx[i, j] = 0
                    continue    
                j_row, j_col = j // height, j % weight
                self._mtx[i, j] = np.abs(i_row - j_row) + np.abs(i_col - j_col)

    def _get_place_shape(self) -> Tuple[int, int]:
        if self.n == 12:
            return (3, 4)
        elif self.n == 30:
            return (5, 6)
        elif self.n == 240:
            return (15, 16)
        elif self.n == 750:
            return (25, 30)
        elif self.n == 2000:
            return (25, 80)
        raise f'unsupported n {self.n}'

class Place2DItem:
    def __init__(self, node: int, x: int, y: int) -> None:
        self.node: int = node
        self.x: int = x
        self.y: int = y

class Place2D(object):
    def __init__(self, 
        mtx: Matrix,
        route_mtx_type=SqauredRouteMtx,
    ) -> None:
        # Indexes setups
        self._x = {}
        self._y = {}
        self._p2c = {}

        self._items: List[Place2DItem] = [None] * mtx.n

        self._route_mtx: RouteMtxBase = route_mtx_type(mtx.n, mtx.dtype)

    @property
    def items(self) -> List[int]: return self._items
    
    @property
    def route_mtx(self) -> np.ndarray: return self._route_mtx.mtx

    def to_nodes(self) -> List[int]: return [item.node for item in self._items]

    def update(self, node: int, pos: int) -> None:
        x, y = self._pos2cord(pos)
        self._items[pos] = Place2DItem(node, x, y)

    def _pos2cord(self, pos: int) -> Tuple[int, int]:
        return self._p2c[pos]
