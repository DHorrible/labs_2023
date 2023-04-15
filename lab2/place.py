import numpy as np

from typing import List, Tuple

from matrix import Matrix

class Place2DItem:
    def __init__(self, node: int, x: int, y: int) -> None:
        self.node: int = node
        self.x: int = x
        self.y: int = y

class Place2D(object):
    def __init__(self, mtx: Matrix) -> None:
        self._mounted = np.array(mtx.mounted, dtype=mtx.dtype)
        
        self._x = {}
        self._y = {}

        self._items: List[Place2DItem] = [None] * mtx.n
        self._p2c = {}

    @property
    def items(self) -> List[int]:
        return self._items
    
    def to_nodes(self) -> List[int]:
        return [item.node for item in self._items]

    def update(self, node: int, pos: int) -> None:
        x, y = self._pos2cord(pos)
        self._items[pos] = Place2DItem(node, x, y)

    def _pos2cord(self, pos: int) -> Tuple[int, int]:
        return self._p2c[pos]
