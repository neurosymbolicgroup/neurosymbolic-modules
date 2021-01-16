from typing import NewType, Tuple, TypeVar, Callable

import numpy as np

from dreamcoder.domains.arc.utils import soft_assert

Color = NewType('Color', int)  # in [-1, 10]
T = TypeVar('T')
S = TypeVar('S')

# handy to have when solving tasks by hand
BLACK: Color = 0
BLUE: Color = 1
RED: Color = 2
GREEN: Color = 3
YELLOW: Color = 4
GREY: Color = 5
PINK: Color = 6
ORANGE: Color = 7
CYAN: Color = 8
MAROON: Color = 9
BACKGROUND_COLOR: Color = -1


class Grid:
    """
    Represents a grid.
    Position is (y, x) where y axis increases downward from 0 at the top.
    """
    def __init__(self, arr):
        assert isinstance(arr, type(np.array([1])))
        assert len(arr.shape) == 2, f"bad arr shape: {arr.shape}"
        assert arr.dtype in [int, np.int32,
                             np.int64], f"bad arr dtype: {arr.dtype}"
        MAX_DIM = 60
        # sometimes it's handy to have a size zero array.
        soft_assert(min(arr.shape) >= 0)
        soft_assert(max(arr.shape) < MAX_DIM)
        # allocates a new array copying the input array
        self.arr = arr.astype(np.int32)

    def __str__(self):
        return str(self.arr)

    def __repr__(self):
        return str(self)

    def __eq__(self, ogrid):
        if hasattr(ogrid, "arr"):
            return np.array_equal(self.arr, ogrid.arr)
        else:
            return False

    @property
    def background_mask(self):
        return self.arr == BACKGROUND_COLOR

    @property
    def foreground_mask(self):
        return self.arr != BACKGROUND_COLOR
