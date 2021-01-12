import numpy as np

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

        ret_arr = np.copy(grid.arr[::scale,::scale])
        return Grid(ret_arr)

    return lambda arg2: deflate(grid=arg1, scale=arg2)


def _kronecker(arg1):
    """Kronecker of arg1.foreground_mask with arg2."""
    def kronecker(grid1: Grid, grid2: Grid) -> Grid:
        ret_arr = np.kron(grid1.foreground_mask, grid2.arr)
        soft_assert(max(ret_arr.shape) < 100)  # prevent memory blowup
        return Grid(ret_arr)

    return lambda arg2: kronecker(grid1=arg1, grid2=arg2)
