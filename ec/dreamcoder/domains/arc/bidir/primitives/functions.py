import numpy as np
import math

from dreamcoder.domains.arc.utils import soft_assert
from dreamcoder.domains.arc.bidir.primitives.types import Color, Grid


def _color_i_to_j(arg1):
    """Changes pixels of color i to color j."""
    def color_i_to_j(grid: Grid, ci: Color, cj: Color) -> Grid:
        out_arr = np.copy(grid.arr)
        out_arr[out_arr == ci] = cj
        return Grid(out_arr)

    return lambda arg2: lambda arg3: color_i_to_j(grid=arg1, ci=arg2, cj=arg3)


def _rotate_ccw(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.arr))


def _rotate_cw(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.arr, k=3))


def _inflate(arg1):
    """
    Does pixel-wise inflation. May want to generalize later.
    Implementation based on https://stackoverflow.com/a/46003970/1337463.
    """
    def inflate(grid: Grid, scale: int) -> Grid:
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

    return lambda arg2: inflate(grid=arg1, scale=arg2)


def _deflate(arg1):
    """
        Given an array and scale, deflates the array in the sense of being
        opposite of inflate.
        
        Input is an array of shape (N x scale, M x scale) and a scale. Assumes
        that array consists of (scale x scale) constant blocks -- i.e is the
        kronecker product of some smaller array and np.ones((scale, scale)).

        Returns the smaller array of shape (N, M).
    """

    def deflate(grid: Grid, scale: int) -> Grid:
        N, M = grid.arr.shape
        soft_assert(N % scale == 0 and M & scale == 0)

        ret_arr = np.copy(grid.arr[::scale, ::scale])
        return Grid(ret_arr)

    return lambda arg2: deflate(grid=arg1, scale=arg2)


def _kronecker(arg1):
    """Kronecker of arg1.foreground_mask with arg2."""
    def kronecker(grid1: Grid, grid2: Grid) -> Grid:
        ret_arr = np.kron(grid1.foreground_mask, grid2.arr)
        soft_assert(max(ret_arr.shape) < 100)  # prevent memory blowup
        return Grid(ret_arr)

    return lambda arg2: kronecker(grid1=arg1, grid2=arg2)


def _crop(Grid: grid) -> Grid:
    """
        Crops to smallest subgrid containing the foreground.
        Based on https://stackoverflow.com/a/48987831/4383594.
    """
    y_range, x_range = np.nonzero(grid.arr == Grid.BACKGROUND_COLOR)
    ret_arr = grid.arr[min(y_range):max(y_range) + 1, 
                       min(x_range):max(x_range) + 1]
    return Grid(ret_arr)


def _set_bg(arg1):
    """
        Sets background color. Alias to color_i_to_j(arg1, color, -1). Note that
        successive set_bg calls build upon each other---the previously set bg
        color does not reappear in the grid.
    """
    return lambda arg2: _map_i_to_j(arg1)(arg2)(-1)


def _size(Grid: grid) -> Grid:
    """ Returns size of grid. """
    return grid.arr.size


def _area(Grid: grid) -> Grid:
    """ Returns number of non-background pixels in grid."""
    return np.count_nonzero(grid.foreground_mask)


def _get_color(Grid: grid) -> Color:
    """ 
        Returns most common color in grid, besides background color -- unless
        grid is blank, in which case returns Grid.BACKGROUND_COLOR. 
    """
    # from https://stackoverflow.com/a/28736715/4383594
    a = np.unique(g.grid, return_counts=True)
    a = zip(*a)
    a = sorted(a, key=lambda t: -t[1])
    a = [x[0] for x in a]
    if a[0] != Grid.BACKGROUND_COLOR or len(a) == 1:
        return a[0]
    return a[1]


def _color_in(arg1):
    """ Colors all non-background pixels to color"""
    def color_in(Grid: grid, Color: color) -> Grid:
        ret_arr = np.copy(grid.arr)
        ret_arr[ret_arr != Grid.BACKGROUND_COLOR] = color
        return Grid(ret_arr)

    return lambda arg2: color_in(grid=arg1, color=arg2)


def _filter_color(arg1):
    """ Sets all pixels not equal to color to background color. """

    def filter_color(Grid: grid, Color: color) -> Grid:
        ret_arr = np.copy(grid.arr)
        ret_arr[ret_arr != color] = Grid.BACKGROUND_COLOR
        return Grid(ret_arr)

    return lambda arg2: filter_color(grid=arg1, color=arg2)


def _top_half(Grid: grid) -> Grid:
    """ Returns top half of grid, including extra row if odd number of rows."""
    r, c = grid.arr.shape
    num_rows = math.ceil(r/2)
    ret_arr = np.copy(grid.arr)[:num_rows]
    return Grid(ret_arr)


def _vflip(Grid: grid) -> Grid:
    """ Flips grid vertically."""
    return Grid(np.copy(np.flip(grid.arr, axis=0)))


def _hflip(Grid: grid) -> Grid:
    """ Flips grid horizontally."""
    return Grid(np.copy(np.flip(grid.arr, axis=1)))


def _empty_grid(arg1):
    """ Returns an empty grid of given shape."""
    def empty_grid(int: height, int: width) -> Grid:
        arr = np.full((height, width), Grid.BACKGROUND_COLOR)
        return Grid(arr)

    return lambda arg2: lambda arg3: empty_grid(height=arg1, width=arg2)
