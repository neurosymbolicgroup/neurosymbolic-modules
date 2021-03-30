import numpy as np
import math
from typing import Tuple, TypeVar, Callable, List, Any, Dict, Type
import typing
from scipy.ndimage import measurements

from bidir.utils import soft_assert
from bidir.primitives.types import Color, Grid, Color

T = TypeVar('T')  # generic type for list ops
S = TypeVar('S')  # another generic type


class Function:
    def __init__(
        self,
        name: str,
        fn: Callable,
        arg_types: List[Type],
        return_type: Type,
    ):
        self.name = name
        self.fn = fn
        self.arg_types = arg_types
        self.arity: int = len(self.arg_types)
        self.return_type: Type = return_type

    def __str__(self):
        return self.name


def make_function(fn: Callable) -> Function:
    """
    Creates a Function for the given function. Infers types from type hints,
    so the op needs to be implemented with type hints.
    """
    types: Dict[str, type] = typing.get_type_hints(fn)
    if len(types) == 0:
        raise ValueError(("Operation provided does not use type hints, "
                          "which we use when choosing ops."))

    return Function(
        name=fn.__name__,
        fn=fn,
        # list of classes, one for each input arg. skip last type (return)
        arg_types=list(types.values())[0:-1],
        return_type=types["return"],
    )


def color_i_to_j(grid: Grid, ci: Color, cj: Color) -> Grid:
    """Changes pixels of color i to color j."""
    out_arr = np.copy(grid.arr)
    out_arr[out_arr == ci.value] = cj.value
    return Grid(out_arr, grid.pos)


def rotate_ccw(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.arr), grid.pos)


def rotate_cw(grid: Grid) -> Grid:
    return Grid(np.rot90(grid.arr, k=3), grid.pos)


def inflate2(grid: Grid) -> Grid:
    return inflate(grid, 2)

def inflate3(grid: Grid) -> Grid:
    return inflate(grid, 3)

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
    return Grid(ret_arr, grid.pos)

def deflate2(grid: Grid) -> Grid:
    return deflate(grid, 2)

def deflate3(grid: Grid) -> Grid:
    return deflate(grid, 3)


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
    return Grid(ret_arr, grid.pos)


def kronecker(grid1: Grid, grid2: Grid) -> Grid:
    """Kronecker of arg1.foreground_mask with arg2."""
    # We want to return an array with BACKGROUND_COLOR for the background,
    # but np.kron uses zero for the background. So we swap with the
    # actual background color and then undo
    inner_arr = np.copy(grid2.arr)

    assert -2 not in inner_arr, 'Invalid -2 color breaks kronecker function'
    inner_arr[inner_arr == 0] = -2

    ret_arr = np.kron(grid1.foreground_mask, inner_arr)

    ret_arr[ret_arr == 0] = Color.BACKGROUND_COLOR.value
    ret_arr[ret_arr == -2] = 0

    soft_assert(max(ret_arr.shape) < 100)  # prevent memory blowup
    return Grid(ret_arr)


def crop(grid: Grid) -> Grid:
    """
    Crops to smallest subgrid containing the foreground.
    If no foreground exists, returns an array of size (0, 0).
    Updates grid's position based on the crop.
    Based on https://stackoverflow.com/a/48987831/4383594.
    """
    if np.all(grid.arr == Color.BACKGROUND_COLOR.value):
        return Grid(np.empty((0, 0), dtype=int))

    y_range, x_range = np.nonzero(grid.arr != Color.BACKGROUND_COLOR.value)
    ret_arr = grid.arr[min(y_range):max(y_range) + 1,
                       min(x_range):max(x_range) + 1]
    pos_delta = min(y_range), min(x_range)
    new_pos = grid.pos[0] + pos_delta[0], grid.pos[1] + pos_delta[1]
    return Grid(ret_arr, pos=new_pos)


def set_bg(grid: Grid, color: Color) -> Grid:
    """
    Sets background color. Alias to color_i_to_j(grid, color, BACKGROUND_COLOR).
    Note that successive set_bg calls build upon each other---the previously
    set bg color does not reappear in the grid.
    """
    return color_i_to_j(grid=grid, ci=color, cj=Color.BACKGROUND_COLOR)


def unset_bg(grid: Grid, color: Color) -> Grid:
    """
    Unsets background color. Alias to color_i_to_j(grid, BACKGROUND_COLOR,
    color).
    """
    return color_i_to_j(grid=grid, ci=Color.BACKGROUND_COLOR, cj=color)


def size(grid: Grid) -> int:
    """Returns the product of the grid's width and height."""
    return grid.arr.size


def area(grid: Grid) -> int:
    """Returns number of non-background pixels in grid."""
    return np.count_nonzero(grid.foreground_mask)  # type: ignore


def width(grid: Grid) -> int:
    """ Returns the width of the grid."""
    return grid.arr.shape[1]


def height(grid: Grid) -> int:
    """ Returns the height of the grid."""
    return grid.arr.shape[0]


def contains_color(grid: Grid, color: Color) -> bool:
    """ Returns whether the color is present in the grid."""
    return color.value in grid.arr


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
    if a[0] != Color.BACKGROUND_COLOR.value or len(a) == 1:
        return Color(a[0])
    return Color(a[1])


def color_in(grid: Grid, color: Color) -> Grid:
    """Colors all non-background pixels to color"""
    ret_arr = np.copy(grid.arr)
    ret_arr[ret_arr != Color.BACKGROUND_COLOR.value] = color.value
    return Grid(ret_arr, grid.pos)


def filter_color(grid: Grid, color: Color) -> Grid:
    """Sets all pixels not equal to color to background color. """
    ret_arr = np.copy(grid.arr)
    ret_arr[ret_arr != color.value] = Color.BACKGROUND_COLOR.value
    return Grid(ret_arr, grid.pos)


def top_half(grid: Grid) -> Grid:
    """Returns top half of grid, including extra row if odd number of rows."""
    r, c = grid.arr.shape
    num_rows = math.ceil(r / 2)
    ret_arr = grid.arr[:num_rows]
    return Grid(ret_arr, grid.pos)


def vflip(grid: Grid) -> Grid:
    """Flips grid vertically."""
    return Grid(np.flip(grid.arr, axis=0), grid.pos)


def hflip(grid: Grid) -> Grid:
    """Flips grid horizontally."""
    return Grid(np.flip(grid.arr, axis=1), grid.pos)


def has_vertical_symmetry(grid: Grid) -> bool:
    return vflip(grid) == grid


def has_horizontal_symmetry(grid: Grid) -> bool:
    return hflip(grid) == grid


def block(height: int, width: int, color: Color) -> Grid:
    """Returns a solid-colored grid of given shape."""
    arr = np.full((height, width), color.value)
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
                                        Color.BACKGROUND_COLOR.value)))

    padded_arrs = [pad(grid.arr, max_width) for grid in grids]
    pos = grids[0].pos if len(grids) > 0 else (0, 0)
    return Grid(np.concatenate(padded_arrs, axis=0), pos)


def hstack(grids: Tuple[Grid, ...]) -> Grid:
    """
    Stacks grids in list horizontally, with first item on left. Pads with
    BACKGROUND_COLOR if necessary.
    """
    max_height = max(grid.arr.shape[0] for grid in grids)

    def pad(arr, height):
        return np.concatenate((arr,
                               np.full((height - arr.shape[0], arr.shape[1]),
                                       Color.BACKGROUND_COLOR.value)))

    padded_arrs = [pad(grid.arr, max_height) for grid in grids]
    pos = grids[0].pos if len(grids) > 0 else (0, 0)
    return Grid(np.concatenate(padded_arrs, axis=1), pos)


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
    return tuple(
        Grid(grid.arr[i:i + 1, :], pos=(i, 0))
        for i in range(grid.arr.shape[0]))


def columns(grid: Grid) -> Tuple[Grid, ...]:
    """Returns a list of columns of the grid."""
    return tuple(
        Grid(grid.arr[:, i:i + 1], pos=(0, i))
        for i in range(grid.arr.shape[1]))


def overlay(grids: Tuple[Grid, ...]) -> Grid:
    """
    Overlays grids on top of each other, with first item on top. Pads to largest
    element's shape, with smaller grids placed in the top left.
    """
    # TODO incorporate positions?
    # if there are positions, uses those.
    height = max(grid.arr.shape[0] for grid in grids)
    width = max(grid.arr.shape[1] for grid in grids)

    out = np.full((height, width), Color.BACKGROUND_COLOR.value)

    def pad(arr, shape):
        pad_height = shape[0] - arr.shape[0]
        pad_width = shape[1] - arr.shape[1]
        return np.pad(arr, ((0, pad_height), (0, pad_width)),
                      'constant',
                      constant_values=Color.BACKGROUND_COLOR.value)

    padded_arrs = [pad(grid.arr, (height, width)) for grid in grids]
    for arr in padded_arrs[::-1]:
        # wherever upper array is not blank, replace with its value
        out[arr != Color.BACKGROUND_COLOR.value] = arr[
            arr != Color.BACKGROUND_COLOR.value]

    return Grid(out)


def overlay_pair(top: Grid, bottom: Grid) -> Grid:
    """
    Overlays two grids, with first on top. Pads to largest argument's shape,
    with the smaller placed in the top left.
    """
    return overlay((top, bottom))


# LIST FUNCTIONS #
# note: many of these are untested.
def map_fn(fn: Callable[[S], T], xs: Tuple[S, ...]) -> Tuple[T, ...]:
    """Maps function onto each element of xs."""
    return tuple(fn(x) for x in xs)


def filter_by_fn(fn: Callable[[T], bool], xs: Tuple[T, ...]) -> Tuple[T, ...]:
    """Returns all elements in xs for which fn is true."""
    return tuple(x for x in xs if fn(x))


def length(xs: Tuple[T, ...]) -> int:
    """Gives the length of xs."""
    return len(xs)


def get(xs: Tuple[T, ...], idx: int) -> T:
    """Gets item at given index of xs."""
    soft_assert(0 <= idx < len(xs))
    return xs[idx]


def reverse(xs: Tuple[T, ...]) -> Tuple[T, ...]:
    """ Reverses the tuple."""
    return xs[::-1]


def sort_by_key(
    xs: Tuple[T, ...],
    keys: Tuple[int, ...],
) -> Tuple[T, ...]:
    """
    Returns first tuple sorted according to corresponding elements in second
    tuple.
    """
    soft_assert(len(xs) == len(keys))

    y = sorted(zip(xs, keys), key=lambda t: t[1])
    # unzips, returns sorted version of xs
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


def order(xs: Tuple[int, ...]) -> Tuple[int, ...]:
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


# def objects(
#     grid: Grid,
#     connect_colors: bool = False,
#     connect_diagonals: bool = True,
# ) -> Tuple[Grid, ...]:
#     def set_bg(grid: Grid) -> Grid:
#         # scipy.ndimage.measurements uses 0 as a background color,
#         # so we map 0 to -2, -1 to 0, then 0 back to -1, and -2 back to zero
#         grid = color_i_to_j(grid, Color(0), Color(-2))
#         grid = color_i_to_j(grid, Color(-1), Color(0))
#         return grid

#     def unset_bg(grid: Grid) -> Grid:
#         grid = color_i_to_j(grid, Color(0), Color(-1))
#         grid = color_i_to_j(grid, Color(-2), Color(0))
#         return grid

#     def mask(arr1, arr2):
#         arr3 = np.copy(arr1)
#         arr3[arr2 == 0] = 0
#         return arr3

#     original_pos = grid.pos

#     def objects_ignoring_colors(
#         grid: Grid,
#         connect_diagonals: bool = False,
#     ) -> List[Grid]:
#         objects = []

#         # if included, this makes diagonally connected components one object.
#         # https://stackoverflow.com/questions/46737409/finding-connected-components-in-a-pixel-array
#         structure = np.ones((3, 3)) if connect_diagonals else None

#         # if items of the same color are separated...then different objects
#         labelled_arr, num_features = measurements.label(grid.arr,
#                                                         structure=structure)
#         for object_i in range(1, num_features + 1):
#             # array with 1 where that object is, 0 elsewhere
#             object_mask = np.where(labelled_arr == object_i, 1, 0)
#             # get the original colors back
#             obj = mask(grid.arr, object_mask)
#             # map back to background and 0 colors, then crop
#             obj = Grid(obj, pos=original_pos)
#             obj = unset_bg(obj)
#             # when cutting out, we automatically set the position, so only need
#             # to add original position
#             obj = crop(obj)
#             objects.append(obj)

#         # print('objects: {}'.format(objects))
#         return objects

#     # print('finding objs: {}'.format(grid))
#     if not connect_colors:
#         separate_color_grids = [
#             filter_color(grid, color) for color in np.unique(grid.arr)
#             if color != 0
#         ]
#         # print('separate_color_grids: {}'.format(separate_color_grids))
#         # print([c for c in np.unique(grid.arr) if c != 0])
#         objects_per_color = [
#             objects_ignoring_colors(color_grid, connect_diagonals)
#             for color_grid in separate_color_grids
#         ]
#         objects = [obj for sublist in objects_per_color for obj in sublist]
#     else:
#         objects = objects_ignoring_colors(grid, connect_diagonals)

#     objects = sorted(objects, key=lambda o: o.pos)
#     return tuple(objects)


def colors(grid: Grid) -> Tuple[Color, ...]:
    """
    Returns a list of colors present in the grid, besides
    Color.BACKGROUND_COLOR.
    """
    return tuple(c for c in set(grid.arr.flatten())
            if c != Color.BACKGROUND_COLOR.value)


# helper function for place_into_grid and place_into_input_grid below
def _place_object(arr: Any, obj: Grid) -> None:
    y, x = obj.pos
    h, w = obj.arr.shape
    g_h, g_w = arr.shape
    # may need to crop the grid for it to fit
    # if negative, crop out the first parts
    o_x, o_y = max(0, -x), max(0, -y)
    # if negative, start at zero instead
    x, y = max(0, x), max(0, y)
    # this also affects the width/height
    w, h = w - o_x, h - o_y
    # if spills out sides, crop out the extra
    w, h = min(w, g_w - x), min(h, g_h - y)
    arr[y:y + h, x:x + w] = obj.arr[o_y:o_y + h, o_x:o_x + w]


def _place_into_grid(arr: Any, objects: Tuple[Grid, ...]) -> Grid:
    for obj in objects:
        _place_object(arr, obj)

    return Grid(arr)


def place_into_grid(objects: Tuple[Grid, ...], grid: Grid) -> Grid:
    blank_grid = np.zeros(grid.arr.shape).astype('int')
    return _place_into_grid(blank_grid, objects)


def place_into_input_grid(objects: Tuple[Grid, ...], grid: Grid) -> Grid:
    grid_copy = np.array(grid.arr)
    return _place_into_grid(grid_copy, objects)
