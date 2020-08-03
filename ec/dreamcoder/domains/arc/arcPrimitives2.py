from dreamcoder.program import *
from dreamcoder.type import arrow, baseType, tint, tlist, t0, t1, t2, tbool
from dreamcoder.task import Task
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

from scipy.ndimage import measurements
import numpy as np

MAX_GRID_LENGTH = 30
MAX_COLOR = 9
MAX_INT = 1

tgrid = baseType("tgrid")
tobject = baseType("tobject")
tpixel = baseType("tpixel")
tcolor = baseType("tcolor")
tinput = baseType("tinput")
tposition = baseType("tposition")

class Grid():
    """
       Represents a grid.
    """
    def __init__(self, grid):
        self.grid = np.array(grid)

    def __str__(self):
        return str(self.grid)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if hasattr(other, "grid"):
            return np.array_equal(self.grid, other.grid)
        else:
            return False

    def absolute_grid(self):
        g = np.zeros((30, 30))
        g[:len(self.grid), :len(self.grid[0])] = self.grid
        return g


class Object(Grid):
    """
       Represents an object. Inherits all of Grid's functions/primitives
    """
    def __init__(self, grid, pos=None, index=0):
        # input the grid with zeros. This turns it into a grid with the
        # background "cut out" and with the position evaluated accordingly

        def cutout(grid):
            x_range, y_range = np.nonzero(grid)

            # for black shapes
            if len(x_range) == 0 and len(y_range) == 0:
                return (0, 0), grid

            pos = min(x_range), min(y_range)
            cut= grid[min(x_range):max(x_range)+1, min(y_range):max(y_range)+1]
            return pos, cut

        pos2, cut = cutout(grid)
        if pos is None: pos = pos2
        super().__init__(cut)
        self.pos = pos
        self.index = index

    def __str__(self):
        return super().__str__() + ', ' + str(self.pos) + ', ix={}'.format(self.index)

    def absolute_grid(self):
        g = np.zeros((30, 30))
        # g = np.zeros(self.pos[0] + len(self.grid), self.pos[1] +
                # len(self.grid[0]))
        g[self.pos[0] : self.pos[0] + len(self.grid), self.pos[1] : self.pos[1]
                + len(self.grid[0])] = self.grid
        return g





class Pixel(Object):
    """
       Represents a single pixel. Inherits all of Object's functions/primitives
    """
    def __init__(self, grid, pos=(0, 0)):
        #TODO how should pixels be initialized?
        super().__init__(grid, pos)
        self.color = grid[0][0]


class Input():
    """
        Combines i/o examples into one input, so that we can synthesize a solution
        which looks at different examples at once
    """
    def __init__(self, input_grid, training_dict):
        self.input_grid = Grid(input_grid)
        # all the examples
        self.grids = [(Grid(ex["input"]), Grid(ex["output"])) for ex in
                training_dict]

    def __str__(self):
        return "i: {}, grids={}".format(self.input_grid, self.grids)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.input_grid == other.input_grid and self.grids == other.grids
        else:
            return False

# list primitives
def _get(l):
    def get(l, i):
        if i < 0 or i > len(l):
            raise ValueError()
        return l[i]

    return lambda i: get(l, i)

def _length(l):
    return len(l)

def _remove_head(l):
    return l[1:]

def _sort(l):
    return lambda f: sorted(l, key=f)

def _map(l):
    return lambda f: [f(x) for x in l]

def _filter(l):
    return lambda f: [x for x in l if f(x)]

def _reverse(l):
    return l[::-1]

def _apply_colors(l_objects):
    return lambda l_colors: [_color_in(o)(c) for (o, c) in zip(l_objects, l_colors)]

def _find_in_list(obj_list):
    def find(obj_list, obj):
        for i, obj2 in enumerate(obj_list):
            if np.array_equal(obj.grid, obj2.grid):
                return i

        return None

    return lambda o: find(obj_list, o)



# grid primitives
def _map_i_to_j(g):
    def map_i_to_j(g, i, j):
        m = np.copy(g.grid)
        m[m==i] = j
        return Grid(m)

    return lambda i: lambda j: map_i_to_j(g, i, j)

def _set_shape(g):
    def set_shape(g, w, h):
        g2 = np.zeros((w, h))
        g2[:len(g.grid), :len(g.grid[0])] = g.grid
        return Grid(g2)

    return lambda s: set_shape(g, s[0], s[1])

def _shape(g):
    return g.grid.shape

def _find_in_grid(g):
    def find(grid, obj):
        for i in range(len(grid) - len(obj) + 1):
            for j in range(len(grid[0]) - len(obj[0]) + 1):
                sub_grid = grid[i: i + len(obj), j : j + len(obj[0])]
                if np.array_equal(obj, sub_grid):
                    return (i, j)
        return None

    return lambda o: find(g.grid, o.grid)

def _filter_color(g):
    return lambda color: Grid(g.grid * (g.grid == color))

def _colors(g):
    # https://stackoverflow.com/questions/15637336/numpy-unique-with-order-preserved
    _, idx = np.unique(g.grid, return_index=True)
    colors = g.grid.flatten()[np.sort(idx)]
    colors = colors.tolist()
    colors.remove(0) # don't want black!
    return colors

def _object(g): #TODO figure out what this needs
    return Object(g.grid, pos=(0,0))

def _pixel2(c):
    return Pixel(np.array([[c]]), pos=(0, 0))

def _pixel(g):
    return lambda i: lambda j: Pixel(g.grid[i:i+1,j:j+1], (i, j))

def _overlay(g):
    return lambda g2: _stack([g, g2])

def _color(g):
    # from https://stackoverflow.com/a/28736715/4383594
    # returns most common color besides black
    a = np.unique(g.grid, return_counts=True)
    a = zip(*a)
    a = sorted(a, key=lambda t: -t[1])
    a = [x[0] for x in a]
    if a[0] != 0 or len(a) == 1:
        return a[0]
    return a[1]

def _objects(g):
    m = np.copy(g.grid)

    # first get all objects
    objects = []

    for color_x in np.unique(m):
        # skip black
        if color_x == 0:
            continue
        # if different colors...then different objects
        # return an array with 1s where that color is, 0 elsewhere
        data_with_only_color_x = np.where(m==color_x, 1, 0) 
        #if items of the same color are separated...then different objects
        data_with_only_color_x_and_object_labels, num_features = measurements.label(data_with_only_color_x)
        for object_i in range(1, num_features + 1):
            # return an array with the appropriate color where that object is, 0 elsewhere
            data_with_only_object_i = np.where(data_with_only_color_x_and_object_labels==object_i, color_x, 0) 
            x_range, y_range = np.nonzero(data_with_only_object_i)
            # position is top left corner of obj
            pos = min(x_range), min(y_range)
            obj = Object(data_with_only_object_i, pos=pos)
            objects.append(obj)

    objects = sorted(objects, key=lambda o: o.pos)
    for i, o in enumerate(objects):
        o.index = i
    return objects

def _pixels(g):
    pixel_grid = [[Pixel(g.grid[i:i+1, j:j+1], (i, j)) 
            for i in range(len(g.grid))]
            for j in range(len(g.grid[0]))]
    # flattens nested list into single list
    return [item for sublist in pixel_grid for item in sublist]

# color primitives

# input primitives
def _input(i): return i.input_grid

def _inputs(i): return [a for (a, b) in i.grids]

def _outputs(i): return [b for (a, b) in i.grids]

def _find_corresponding(i):
    # object corresponding to object - working with lists of objects
    def location_in_input(inp, o):
        for i, input_example in enumerate(_inputs(inp)):
            objects = _objects(input_example)
            location = _find_in_list(objects)(o)
            if location is not None:
                return i, location
        return None

    def find(inp, o):
        location = location_in_input(inp, o)
        if location is None: raise ValueError()
        out = _get(_objects(_get(_outputs(inp))(location[0])))(location[1])
        # make the position of the newly mapped equal to the input positions
        out.pos = o.pos
        return out

    return lambda o: find(i, o)

# list consolidation
def _vstack(l):
    # stacks list of grids atop each other based on dimensions
    # TODO won't work if they have different dimensions
    if not np.all([len(l[0].grid[0]) == len(x.grid[0]) for x in l]):
        raise ValueError()
    l = [x.grid for x in l]
    return Grid(np.concatenate(l, axis=0))

def _hstack(l):
    # stacks list of grids horizontally based on dimensions
    # TODO won't work if they have different dimensions
    if not np.all([len(l[0].grid) == len(x.grid) for x in l]):
        raise ValueError()
    return Grid(np.concatenate(l, axis=1))

def _positionless_stack(l):
    # doesn't use positions, just absolute object shape + overlay
    grid = np.zeros((30, 30))
    for g in l:
        # mask later additions
        grid += g.grid * (grid == 0)

    # get rid of extra shape -- needed?
    grid = Object(grid, pos=(0, 0))

    return Grid(grid.grid)

def _stack(l):
    # stacks based on positions atop each other, masking first to last
    grid = np.zeros((30, 30))
    for g in l:
        # mask later additions
        grid += g.absolute_grid() * (grid == 0)

    # get rid of extra shape -- needed?
    grid = Object(grid, pos=(0, 0))

    return Grid(grid.grid)



# boolean primitives
def _and(a): return lambda b: a and b
def _or(a): return lambda b: a or b
def _not(a): return not a
def _ite(a): return lambda b: lambda c: b if a else c 
def _eq(a): return lambda b: a == b

# object primitives
def _index(o): return o.index
def _position(o): return o.pos
def _x(o): return o.pos[0]
def _y(o): return o.pos[1]
def _size(o): return o.grid.size
def _area(o): return np.sum(o.grid != 0)

def _color_in(o):
    def color_in(o, c):
        grid = o.grid
        grid[grid != 0] = c
        return Object(grid, o.pos, o.index)

    return lambda c: color_in(o, c)

# pixel primitives


# misc primitives
def _inflate(o):
    # currently does pixel-wise inflation. may want to generalize later
    def inflate(o, scale):
        # scale is 1, 2, 3, maybe 4
        x, y = o.grid.shape
        shape = (x*scale, y*scale)
        grid = np.zeros(shape)
        for i in range(len(o.grid)):
            for j in range(len(o.grid[0])):
                grid[scale * i : scale * (i + 1),
                     scale * j : scale * (j + 1)] = o.grid[i,j]

        return Object(grid)

    return lambda inflate_factor: inflate(o, inflate_factor)

def _half(o):
    def half(grid, way): #TODO add position
        if way == 1:
            # top half
            return grid[0:int(len(grid)/2), :]
        if way == 2:
            # bottom half
            return grid[int(len(grid)/2):, :]
        if way == 3:
            # left
            return grid[:, 0:int(len(grid[0])/2)]
        else:
            if way != 4:
                raise IndexError()
            # right
            return grid[:, int(len(grid[0])/2):]

    return lambda i: half(o, i)


## making the actual primitives

colors = [
    Primitive(str(i), tcolor, i) for i in range(0, MAX_COLOR + 1)
    ]
ints = [
    # Primitive(str(i), tint, i) for i in range(0, MAX_INT + 1)
    ]
bools = [
    # Primitive("True", tbool, True),
    # Primitive("False", tbool, False)
    ]

list_primitives = [
    Primitive("get", arrow(tlist(t0), t0), _get),
    Primitive("length", arrow(tlist(t0), tint), _length),
    Primitive("remove_head", arrow(tlist(t0), t0), _remove_head),
    Primitive("sort", arrow(tlist(t0), tlist(t0)), _sort),
    Primitive("map", arrow(tlist(t0), arrow(t0, t1), tlist(t1)), _map),
    Primitive("filter", arrow(tlist(t0), arrow(t0, tbool), tlist(t0)), _filter),
    Primitive("reverse", arrow(tlist(t0), tlist(t0)), _reverse),
    Primitive("apply_colors", arrow(tlist(tgrid), tlist(tcolor)), _apply_colors)
    ]

grid_primitives = [
    # Primitive("find_in_list", arrow(tlist(tgrid), tint), _find_in_list),
    # Primitive("find_in_grid", arrow(tgrid, tgrid, tposition), _find_in_grid),
    Primitive("filter_color", arrow(tgrid, tgrid), _filter_color),
    Primitive("colors", arrow(tgrid, tlist(tcolor)), _colors),
    Primitive("object", arrow(tgrid, tgrid), _object),
    Primitive("pixel2", arrow(tcolor, tgrid), _pixel2),
    # Primitive("pixel", arrow(tint, tint, tgrid), _pixel),
    # Primitive("overlay", arrow(tgrid, tgrid, tgrid), _overlay),
    Primitive("color", arrow(tgrid, tgrid), _color),
    Primitive("objects", arrow(tgrid, tlist(tgrid)), _objects),
    Primitive("pixels", arrow(tgrid, tlist(tgrid)), _pixels),
    # Primitive("set_shape", arrow(tgrid, tposition, tgrid), _set_shape),
    # Primitive("shape", arrow(tgrid, tposition), _shape)
    ]

input_primitives = [
    Primitive("input", arrow(tinput, tgrid), _input),
    Primitive("inputs", arrow(tinput, tlist(tgrid)), _inputs),
    Primitive("outputs", arrow(tinput, tlist(tgrid)), _outputs),
    Primitive("find_corresponding", arrow(tinput, tgrid, tgrid), _find_corresponding)
    ]

list_consolidation = [
    Primitive("vstack", arrow(tlist(tgrid), tgrid), _vstack),
    Primitive("hstack", arrow(tlist(tgrid), tgrid), _hstack),
    Primitive("positionless_stack", arrow(tlist(tgrid), tgrid), _positionless_stack),
    Primitive("stack", arrow(tlist(tgrid), tgrid), _stack),
    ]

boolean_primitives = [
    # Primitive("and", arrow(tbool, tbool, tbool), _and),
    # Primitive("or", arrow(tbool, tbool, tbool), _or),
    # Primitive("not", arrow(tbool, tbool), _not),
    # Primitive("ite", arrow(tbool, t0, t0, t0), _ite),
    # Primitive("eq", arrow(t0, t0, tbool), _eq)
    ]

object_primitives = [
    Primitive("index", arrow(tgrid, tint), _index),
    # Primitive("position", arrow(tgrid, tposition), _position),
    # Primitive("x", arrow(tgrid, tint), _x),
    # Primitive("y", arrow(tgrid, tint), _y),
    Primitive("color_in", arrow(tgrid, tcolor, tgrid), _color_in),
    # Primitive("size", arrow(tgrid, tint), _size),
    Primitive("area", arrow(tgrid, tint), _area)
    ]

misc_primitives = [
    Primitive("inflate", arrow(tgrid, tint, tgrid), _inflate),
    # Primitive("half", arrow(tgrid, tint, tgrid), _half)
    ]

primitives = colors + ints + bools + list_primitives + grid_primitives + input_primitives + list_consolidation + boolean_primitives + object_primitives +  misc_primitives
