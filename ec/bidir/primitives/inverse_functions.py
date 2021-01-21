from typing import Tuple
from bidir.primitives.types import Color, Grid
import numpy as np


def cond_assert(condition: bool, message: str = None) -> None:
    """
    Used for asserting the correct numpy and locations of None's in the inputs
    for a conditional inverse function.
    """
    if not condition:
        raise ValueError(message)

def inv_assert(condition: bool, message: str = None) -> None:
    """
    Used for asserting qualities of the output provided to an inverse function,
    or the inputs provided to a conditional inverse function.
    """
    if not condition:
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
    cond_assert(sum(i is None for i in in_grids) == 1)
    top, bottom = in_grids
    out_h, out_w = out.arr.shape

    def split_at_row(arr, row):
        return arr[:row], arr[row:]

    if top is None:
        bottom_h, bottom_w = bottom.arr.shape
        inv_assert(out_w == bottom_w)
        out_top, out_bottom = split_at_row(out.arr, out_h - bottom_h)
        inv_assert(np.array_equal(out_bottom, bottom.arr))
        top = Grid(out_top)
    else:  # bottom is None
        top_h, top_w = top.arr.shape
        inv_assert(out_w == top_w)
        out_top, out_bottom = split_at_row(out.arr, top_h)
        inv_assert(np.array_equal(out_top, top.arr))
        bottom = Grid(out_bottom)

    return (top, bottom)


def hstack_pair_cond_inv(
    out: Grid,
    in_grids: Tuple[Grid, Grid],
) -> Tuple[Grid, Grid]:
    """
    Conditional inverse of hstack_pair.
    """
    cond_assert(sum(i is None for i in in_grids) == 1)
    left, right = in_grids
    out_h, out_w = out.arr.shape

    def split_at_col(arr, col):
        return arr[:, :col], arr[:, col:]

    if left is None:
        right_h, right_w = right.arr.shape
        inv_assert(out_h == right_h)
        out_left, out_right = split_at_col(out.arr, out_w - right_w)
        inv_assert(np.array_equal(out_right, right.arr))
        left = Grid(out_left)
    else:  # right is None
        left_h, left_w = left.arr.shape
        inv_assert(out_h == left_h)
        out_left, out_right = split_at_col(out.arr, left_w)
        inv_assert(np.array_equal(out_left, left.arr))
        right = Grid(out_right)

    return (left, right)


def block_inv(grid: Grid) -> Tuple[Color, int, int]:
    """
    Exact inverse of block().
    """
    color = grid.arr[0, 0]
    H, W = grid.arr.shape
    inv_assert(np.all(grid.arr == color))
    return (H, W, color)
