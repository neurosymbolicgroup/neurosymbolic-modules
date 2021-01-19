import numpy as np
import math
from typing import Tuple, TypeVar, Callable

from bidir.utils import soft_assert
from bidir.primitives.types import (
    Color,
    Grid,
    BACKGROUND_COLOR,
)

T = TypeVar('T')  # generic type for list ops
S = TypeVar('S')  # another generic type


def color_i_to_j(grid: Grid, ci: Color, cj: Color) -> Grid:
    """Changes pixels of color i to color j."""
    out_arr = np.copy(grid.arr)
    out_arr[out_arr == ci] = cj
    return Grid(out_arr)


def rotate_ccw(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.arr))


def rotate_cw(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.arr, k=3))


def inflate(grid: Grid, scale: int) -> Grid:
    """
    Does pixel-wise inflation. May want to generalize later.
    Implementation based on https://stackoverflow.com/a/46003970/1337463.
    """
    soft_assert(scale <= 10)  # scale is 1, 2, 3, maybe 4
    soft_assert(scale >= 0)
    ret_arr = np.kron(
        grid.arr,
        np.ones(
            (scale, scale),
            dtype=grid.arr.dtype,
        ),
    )
    return Grid(ret_arr)


def deflate(grid: Grid, scale: int) -> Grid:
    """
    Given an array and scale, deflates the array in the sense of being
    opposite of inflate.

    Input is an array of shape (N x scale, M x scale) and a scale. Assumes
    that array consists of (scale x scale) constant blocks -- i.e is the
    kronecker product of some smaller array and np.ones((scale, scale)).

    Returns the smaller array of shape (N, M).
    """
    N, M = grid.arr.shape
    soft_assert(N % scale == 0 and M % scale == 0)

    ret_arr = grid.arr[::scale, ::scale]
    return Grid(ret_arr)


def kronecker(grid1: Grid, grid2: Grid) -> Grid:
    """Kronecker of arg1.foreground_mask with arg2."""
    # We want to return an array with BACKGROUND_COLOR for the background,
    # but np.kron uses zero for the background. So we swap with the
    # actual background color and then undo
    inner_arr = np.copy(grid2.arr)

    assert -2 not in inner_arr, 'Invalid -2 color breaks kronecker function'
    inner_arr[inner_arr == 0] = -2

    ret_arr = np.kron(grid1.foreground_mask, inner_arr)

    ret_arr[ret_arr == 0] = BACKGROUND_COLOR
    ret_arr[ret_arr == -2] = 0

    soft_assert(max(ret_arr.shape) < 100)  # prevent memory blowup
    return Grid(ret_arr)


def crop(grid: Grid) -> Grid:
    """
    Crops to smallest subgrid containing the foreground.
    If no foreground exists, returns an array of size (0, 0).
    Based on https://stackoverflow.com/a/48987831/4383594.
    """
    if np.all(grid.arr == BACKGROUND_COLOR):
        return Grid(np.empty((0, 0), dtype=int))

    y_range, x_range = np.nonzero(grid.arr != BACKGROUND_COLOR)
    ret_arr = grid.arr[min(y_range):max(y_range) + 1,
                       min(x_range):max(x_range) + 1]
    return Grid(ret_arr)


def set_bg(grid: Grid, color: Color) -> Grid:
    """
    Sets background color. Alias to color_i_to_j(grid, color, BACKGROUND_COLOR).
    Note that successive set_bg calls build upon each other---the previously
    set bg color does not reappear in the grid.
    """
    return color_i_to_j(grid=grid, ci=color, cj=BACKGROUND_COLOR)


def unset_bg(grid: Grid, color: Color):
    """
    Unsets background color. Alias to color_i_to_j(grid, BACKGROUND_COLOR,
    color).
    """
    return color_i_to_j(grid=grid, ci=BACKGROUND_COLOR, cj=color)


def size(grid: Grid) -> int:
    """Returns the product of the grid's width and height."""
    return grid.arr.size


def area(grid: Grid) -> int:
    """Returns number of non-background pixels in grid."""
    return np.count_nonzero(grid.foreground_mask)  # type: ignore


def get_color(grid: Grid) -> Color:
    """
    Returns most common color in grid, besides background color -- unless
    grid is blank, in which case returns BACKGROUND_COLOR.
    """
    # from https://stackoverflow.com/a/28736715/4383594
    a = np.unique(grid.arr, return_counts=True)
    a = zip(*a)
    a = sorted(a, key=lambda t: -t[1])
    a = [x[0] for x in a]
    if a[0] != BACKGROUND_COLOR or len(a) == 1:
        return a[0]
    return a[1]


def color_in(grid: Grid, color: Color) -> Grid:
    """Colors all non-background pixels to color"""
    ret_arr = np.copy(grid.arr)
    ret_arr[ret_arr != BACKGROUND_COLOR] = color
    return Grid(ret_arr)


def filter_color(grid: Grid, color: Color) -> Grid:
    """Sets all pixels not equal to color to background color. """
    ret_arr = np.copy(grid.arr)
    ret_arr[ret_arr != color] = BACKGROUND_COLOR
    return Grid(ret_arr)


def top_half(grid: Grid) -> Grid:
    """Returns top half of grid, including extra row if odd number of rows."""
    r, c = grid.arr.shape
    num_rows = math.ceil(r / 2)
    ret_arr = grid.arr[:num_rows]
    return Grid(ret_arr)


def vflip(grid: Grid) -> Grid:
    """Flips grid vertically."""
    return Grid(np.flip(grid.arr, axis=0))


def hflip(grid: Grid) -> Grid:
    """Flips grid horizontally."""
    return Grid(np.flip(grid.arr, axis=1))


def empty_grid(height: int, width: int) -> Grid:
    """Returns an empty grid of given shape."""
    arr = np.full((height, width), BACKGROUND_COLOR)
    return Grid(arr)


def vstack(grids: Tuple[Grid, ...]) -> Grid:
    """
    Stacks grids in list vertically, with first item on top. Pads with
    BACKGROUND_COLOR if necessary.
    """
    max_width = max(grid.arr.shape[1] for grid in grids)

    def pad(arr, width):
        return np.column_stack((arr,
                                np.full((arr.shape[0], width - arr.shape[1]),
                                        BACKGROUND_COLOR)))

    padded_arrs = [pad(grid.arr, max_width) for grid in grids]
    return Grid(np.concatenate(padded_arrs, axis=0))


def hstack(grids: Tuple[Grid, ...]) -> Grid:
    """
    Stacks grids in list horizontally, with first item on left. Pads with
    BACKGROUND_COLOR if necessary.
    """
    max_height = max(grid.arr.shape[0] for grid in grids)

    def pad(arr, height):
        return np.concatenate((arr,
                               np.full((height - arr.shape[0], arr.shape[1]),
                                       BACKGROUND_COLOR)))

    padded_arrs = [pad(grid.arr, max_height) for grid in grids]
    return Grid(np.concatenate(padded_arrs, axis=1))


def vstack_pair(top: Grid, bottom: Grid) -> Grid:
    """
    Stacks first argument above second argument, padding with
    BACKGROUND_COLOR if necessary.
    """
    return vstack((top, bottom))


def hstack_pair(left: Grid, right: Grid) -> Grid:
    """
    Stacks first argument left of second argument, padding with
    BACKGROUND_COLOR if necessary.
    """
    return hstack((left, right))


def rows(grid: Grid) -> Tuple[Grid, ...]:
    """Returns a list of rows of the grid."""
    return tuple(Grid(grid.arr[i:i + 1, :]) for i in range(grid.arr.shape[0]))


def columns(grid: Grid) -> Tuple[Grid, ...]:
    """Returns a list of columns of the grid."""
    return tuple(Grid(grid.arr[:, i:i + 1]) for i in range(grid.arr.shape[1]))


def overlay(grids: Tuple[Grid, ...]) -> Grid:
    """
    Overlays grids on top of each other, with first item on top. Pads to largest
    element's shape, with smaller grids placed in the top left.
    """
    # if there are positions, uses those.
    height = max(grid.arr.shape[0] for grid in grids)
    width = max(grid.arr.shape[1] for grid in grids)

    out = np.full((height, width), BACKGROUND_COLOR)

    def pad(arr, shape):
        pad_height = arr.shape[0] - shape[0]
        pad_width = arr.shape[1] - shape[1]
        return np.pad(arr, ((0, pad_height), (0, pad_width)),
                      'constant',
                      constant_values=BACKGROUND_COLOR)

    padded_arrs = [pad(grid.arr, (height, width)) for grid in grids]
    for arr in padded_arrs[::-1]:
        # wherever upper array is not blank, replace with its value
        out[arr != BACKGROUND_COLOR] = arr[arr != BACKGROUND_COLOR]

    return Grid(out)


def overlay_pair(top: Grid, bottom: Grid) -> Grid:
    """
    Overlays two grids, with first on top. Pads to largest argument's shape,
    with the smaller placed in the top left.
    """
    return overlay((top, bottom))


######## LIST FUNCTIONS ###########
# note: many of these are untested.
def map_fn(f: Callable[[S], T], xs: Tuple[S, ...]) -> Tuple[T, ...]:
    """Maps function onto each element of xs."""
    return tuple(f(x) for x in xs)


def filter_by_fn(f: Callable[[T], bool], xs: Tuple[T, ...]) -> Tuple[T, ...]:
    """Returns all elements in xs for which f is true."""
    return tuple(x for x in xs if f(x))


def length(xs: Tuple[T, ...]) -> int:
    """Gives the length of xs."""
    return len(xs)


def get(xs: Tuple[T, ...], idx: int) -> T:
    """Gets item at given index of xs."""
    soft_assert(0 <= idx < len(xs), 'index out of range')
    return xs[idx]


def sort_by_key(
    tuple1: Tuple[T, ...],
    tuple2: Tuple[int, ...],
) -> Tuple[T, ...]:
    """
    Returns first tuple sorted according to corresponding elements in second
    tuple.
    """
    soft_assert(len(tuple1) == len(tuple2), 'list lengths must be equal')

    y = sorted(zip(tuple1, tuple2), key=lambda t: t[1])
    # unzips list, returns sorted version of list1
    return tuple(list(zip(*y))[0])


def frequency(xs: Tuple[T, ...]) -> Tuple[int, ...]:
    """
    Returns how often each item occurs in the tuple. That is, if f(x) returns
    how frequent x is in the tuple, returns [f(x) for x in l].
    """
    dict_freq = {x: 0 for x in xs}
    for x in xs:
        dict_freq[x] += 1

    return tuple(dict_freq[x] for x in xs)


def order(xs):
    """
    Takes the unique numbers in the tuple, sorts them. Then if f(x) returns the
    index of a number in this sorted list, returns [f(x) for x in l].
    For example, if the input is [4, 1, 1, 2], returns [2, 0, 0, 1].
    """
    # remove duplicates
    arr = np.asarray(list(set(xs)))
    result = np.argsort(arr)
    order_dict = {result[i]: i for i in range(len(result))}
    return tuple(order_dict[x] for x in xs)
