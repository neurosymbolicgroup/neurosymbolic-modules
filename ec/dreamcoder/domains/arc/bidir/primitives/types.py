from typing import NewType

import numpy as np

from dreamcoder.domains.arc.utils import soft_assert

Color = NewType('Color', int)  # in [-1, 10]

# handy when solving tasks by hand
COLORS = {'black': 0,
          'blue': 1,
          'red': 2,
          'green': 3,
          'yellow': 4,
          'grey': 5,
          'pink': 6,
          'orange': 7,
          'cyan': 8,
          'maroon': 9,
          'background': -1}

class Grid:
    """
    Represents a grid.
    Position is (y, x) where y axis increases downward from 0 at the top.
    """
    BACKGROUND_COLOR: Color = -1

    def __init__(self, arr):
        assert isinstance(arr, type(np.array([1])))
        assert len(arr.shape) == 2, f"bad arr shape: {arr.shape}"
        assert arr.dtype in [int, np.int32,
                             np.int64], f"bad arr dtype: {arr.dtype}"
        MAX_DIM = 60
        soft_assert(min(arr.shape) > 0)
        soft_assert(max(arr.shape) < MAX_DIM)
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
        return self.arr == Grid.BACKGROUND_COLOR

    @property
    def foreground_mask(self):
        return self.arr != Grid.BACKGROUND_COLOR
