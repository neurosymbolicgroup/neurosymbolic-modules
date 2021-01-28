from typing import Tuple, Any
import numpy as np

from bidir.primitives.types import Color, Grid
import bidir.primitives.functions as F


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
    return F.deflate(g, i)


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


def block_inv(grid: Grid) -> Tuple[int, int, Color]:
    """
    Exact inverse of block().
    """
    color = grid.arr[0, 0]
    H, W = grid.arr.shape
    inv_assert_equal(grid.arr, np.full((H, W), color))
    return (H, W, color)

def kronecker_cond_inv(out_grid: Grid, template_height: int, template_width: int) -> Tuple[Grid, Grid]:
    """
    Conditional inverse of kronecker() given the dimensions of the template
    """
    M,N = out_grid.arr.shape
    M1,N1 = template_height, template_width
    M2 = M//template_height
    N2 = N//template_width
    reshape_out = np.zeros((M1*N1, M2*N2))
    col_idxs = []
    for i in range(M2):
        for j in range(N2):
            reshape_out[:,i*N2 + j] = out_grid[i*M1:(i+1)*M1, j*N1:(j+1)*N1].ravel()
            if np.linalg.norm(reshape_out[:,i*N2+j],2) > 0:
                col_idxs.append(i*N2+j)
    for c in col_idxs:
        inv_assert_equal(reshape_out[:,c], reshape_out[:,col_idxs[0]], message="out_grid not a kronecker product")

    fg_mask = np.zeros((M2**N2))
    fg_mask[col_idxs] = 1
    fg_mask = fg_mask.reshape(M2,N2)
    template = reshape_out[:,col_idxs[0]].reshape(M1,N1)
    return Grid(template), Grid(fg_mask)

def color_i_to_j_cond_inv(grid:Grid, ci: Color, cj: Color) -> Grid:
    return F.color_i_to_j(grid, cj, ci)

def sort_by_key_cond_inv(xs: Tuple, ys: Tuple) -> Tuple[int,...]:
    """
    Conditional inverse of sort_by_key() that returns the permutation of
    elements in the input list to obtain the output list
    """
    entity_dict = {}
    for i,x in enumerate(xs):
        entity_dict[x] = i
    try:
        sort_key = [d[y] for y in ys]
    except KeyError:
        raise Exception("Output is not a sorted version of input")
    return sort_key

def overlay_pair_cond_inv(
    out: Grid,
    in_grids: Tuple[Grid, Grid],
) -> Tuple[Grid, Grid]:
    """
    Conditional inverse of overlay_pair.
    """
    cond_assert(sum(i is None for i in in_grids) == 1, in_grids)
    top, bottom = in_grids

    def pad(arr, shape):
        pad_height = shape[0] - arr.shape[0]
        pad_width = shape[1] - arr.shape[1]
        return np.pad(arr, ((0, pad_height), (0, pad_width)),
                      'constant',
                      constant_values=COLORS.BACKGROUND_COLOR)

    if top is None:
        bottom = Grid(pad(bottom, out.shape))
        cond_assert(np.sum(bottom.arr == out.arr) > 0 , (out, bottom))
        top_arr = np.copy(out.arr)
        top_arr[bottom.arr==out.arr] = COLORS.BACKGROUND_COLOR
        top = Grid(top_arr)
    else:  # bottom is None
        top = Grid(pad(top, out.shape))
        cond_assert(np.sum(top.arr == out.arr) > 0 , (out, top))
        bottom_arr = np.copy(out.arr)
        bottom_arr[top.arr == out.arr] = COLORS.BACKGROUND_COLOR
        bottom = Grid(bottom_arr)

    return (top, bottom)
