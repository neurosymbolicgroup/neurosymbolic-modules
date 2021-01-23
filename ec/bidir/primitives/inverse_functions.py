from typing import Tuple, Any
from bidir.primitives.types import Color, Grid
import numpy as np


def cond_assert(condition: bool, args_given: Tuple) -> None:
    """
    Used for asserting the correct numpy and locations of None's in the inputs
    for a conditional inverse function.
    """
    message = f"incorrect input args for conditional assert: {args_given}"
    if not condition:
        raise ValueError(message)


def inv_assert_equal(first: Any, second: Any, message: str = "") -> None:
    """
    Used for asserting qualities of the output provided to an inverse function,
    or the inputs provided to a conditional inverse function.
    """
    message = (f"Expected these two to be equal:\n"
               f"First: \t{first}\n"
               f"Second:\t{second}\n"
               f"{message}")

    if (isinstance(first, np.ndarray) and isinstance(second, np.ndarray)):
        if not np.array_equal(first, second):
            raise ValueError(message)
    elif first != second:
        raise ValueError(message)


def inflate_cond_inv(g: Grid, i: int) -> Grid:
    return deflate(g, i)


def vstack_pair_cond_inv(
    out: Grid,
    in_grids: Tuple[Grid, Grid],
) -> Tuple[Grid, Grid]:
    """
    Conditional inverse of vstack_pair.
    """
    cond_assert(sum(i is None for i in in_grids) == 1, in_grids)
    top, bottom = in_grids
    out_h, out_w = out.arr.shape

    def split_at_row(arr, row):
        return arr[:row], arr[row:]

    if top is None:
        bottom_h, bottom_w = bottom.arr.shape
        inv_assert_equal(out_w, bottom_w)
        out_top, out_bottom = split_at_row(out.arr, out_h - bottom_h)
        inv_assert_equal(out_bottom, bottom.arr)
        top = Grid(out_top)
    else:  # bottom is None
        top_h, top_w = top.arr.shape
        inv_assert_equal(out_w, top_w)
        out_top, out_bottom = split_at_row(out.arr, top_h)
        inv_assert_equal(out_top, top.arr)
        bottom = Grid(out_bottom)

    return (top, bottom)


def hstack_pair_cond_inv(
    out: Grid,
    in_grids: Tuple[Grid, Grid],
) -> Tuple[Grid, Grid]:
    """
    Conditional inverse of hstack_pair.
    """
    cond_assert(sum(i is None for i in in_grids) == 1, in_grids)
    left, right = in_grids
    out_h, out_w = out.arr.shape

    def split_at_col(arr, col):
        return arr[:, :col], arr[:, col:]

    if left is None:
        right_h, right_w = right.arr.shape
        inv_assert_equal(out_h, right_h)
        out_left, out_right = split_at_col(out.arr, out_w - right_w)
        inv_assert_equal(out_right, right.arr)
        left = Grid(out_left)
    else:  # right is None
        left_h, left_w = left.arr.shape
        inv_assert_equal(out_h, left_h)
        out_left, out_right = split_at_col(out.arr, left_w)
        inv_assert_equal(out_left, left.arr)
        right = Grid(out_right)

    return (left, right)


def block_inv(grid: Grid) -> Tuple[Color, int, int]:
    """
    Exact inverse of block().
    """
    color = grid.arr[0, 0]
    H, W = grid.arr.shape
    inv_assert_equal(grid.arr, np.full((H, W), color))
    return (H, W, color)
