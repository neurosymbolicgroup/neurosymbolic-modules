import numpy as np

from dreamcoder.domains.arc.utils import soft_assert
from dreamcoder.domains.arc.bidir.primitives.types import Color, Grid


def color_i_to_j_(arg1):
    """Changes pixels of color i to color j."""
    def color_i_to_j_(grid: Grid, ci: Color, cj: Color):
        out_arr = np.copy(grid.arr)
        out_arr[out_arr == ci] = cj
        return Grid(out_arr)

    return lambda arg2: lambda arg3: color_i_to_j_(grid=arg1, ci=arg2, cj=arg3)


def inflate_(arg1: Grid):
    """
    Does pixel-wise inflation. May want to generalize later.
    Implementation based on https://stackoverflow.com/a/46003970/1337463.
    """
    def inflate(grid: Grid, scale: int):
        soft_assert(scale <= 10)  # scale is 1, 2, 3, maybe 4
        ret_arr = np.kron(
            grid.arr,
            np.ones(
                (scale, scale),
                dtype=grid.arr.dtype,
            ),
        )
        return Grid(ret_arr)

    return lambda arg2: inflate(grid=arg1, scale=arg2)


def kronecker_(arg1):
    """Kronecker of arg1.foreground_mask with arg2."""
    def kronecker(grid1: Grid, grid2: Grid):
        ret_arr = np.kron(grid1.foreground_mask, grid2.arr)
        soft_assert(max(ret_arr.shape) < 100)  # prevent memory blowup
        return Grid(ret_arr)

    return lambda arg2: kronecker(grid1=arg1, grid2=arg2)
