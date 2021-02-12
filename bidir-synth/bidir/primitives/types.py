import enum
from typing import Tuple

import numpy as np

from bidir.utils import soft_assert



class Color(enum.Enum):
    """
    Colors present in ARC grids.
    """
    BACKGROUND_COLOR = -1
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GREY = 5
    PINK = 6
    ORANGE = 7
    CYAN = 8
    MAROON = 9


class Grid:
    """
    Represents a grid.
    Position is (y, x) where y axis increases downward from 0 at the top.
    To make it easier to implement various grid operations, we store the grid
    as a numpy array of ints, not colors.
    """
    def __init__(self, arr: np.ndarray, pos: Tuple[int, int] = (0, 0)):
        assert isinstance(arr, type(np.array([1])))
        assert len(arr.shape) == 2, f"bad arr shape: {arr.shape}"
        assert arr.dtype in [int, np.int32,
                             np.int64], f"bad arr dtype: {arr.dtype}"
        MAX_DIM = 60
        # sometimes it's handy to have a size zero array.
        soft_assert(min(arr.shape) >= 0)
        soft_assert(max(arr.shape) < MAX_DIM)
        # allocates a new array copying the input array
        self.arr: np.ndarray = arr.astype(np.int32)
        self.pos = pos

    def __str__(self) -> str:
        # return f"({self.arr}, {self.pos})"
        return str(self.arr)

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, ogrid) -> bool:
        if hasattr(ogrid, "arr"):
            return np.array_equal(self.arr, ogrid.arr)
        else:
            return False

    def __hash__(self):
        # see https://stackoverflow.com/a/16592241/4383594
        return hash(str(self.arr))

    @property
    def background_mask(self) -> np.ndarray:
        return self.arr == Color.BACKGROUND_COLOR.value

    @property
    def foreground_mask(self) -> np.ndarray:
        return self.arr != Color.BACKGROUND_COLOR.value
