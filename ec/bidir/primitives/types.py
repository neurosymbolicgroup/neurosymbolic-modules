from typing import NewType, Tuple

import numpy as np

from bidir.utils import soft_assert

Color = NewType('Color', int)  # in [-1, 10]


class COLORS:
    """
    Constaints constants for colors.
    Handy to have when solving tasks by hand.
    """
    BACKGROUND_COLOR = Color(-1)
    BLACK = Color(0)
    BLUE = Color(1)
    RED = Color(2)
    GREEN = Color(3)
    YELLOW = Color(4)
    GREY = Color(5)
    PINK = Color(6)
    ORANGE = Color(7)
    CYAN = Color(8)
    MAROON = Color(9)

    ALL_COLORS = tuple(Color(i) for i in range(-1, 10))

    def name_of(col: Color) -> str:
        colors = [
            'Background',
            'Black',
            'Blue',
            'Red',
            'Green',
            'Yellow',
            'Grey',
            'Pink',
            'Orange',
            'Cyan',
            'Maroon',
        ]

        return colors[col - 1]


class Grid:
    """
    Represents a grid.
    Position is (y, x) where y axis increases downward from 0 at the top.
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
        return self.arr == COLORS.BACKGROUND_COLOR

    @property
    def foreground_mask(self) -> np.ndarray:
        return self.arr != COLORS.BACKGROUND_COLOR
