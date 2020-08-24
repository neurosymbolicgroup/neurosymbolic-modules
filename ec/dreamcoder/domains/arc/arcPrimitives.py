from dreamcoder.program import *
from dreamcoder.type import arrow, baseType, tint, tlist, t0, t1, t2, tbool
from dreamcoder.task import Task
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

from scipy.ndimage import measurements
from math import sin,cos,radians,copysign
import numpy as np

MAX_GRID_LENGTH = 30
MAX_COLOR = 9
MAX_INT = 9

tgrid = baseType("tgrid")
tobject = baseType("tobject")
tpixel = baseType("tpixel")
tcolor = baseType("tcolor")
tdir = baseType("tdir")
tinput = baseType("tinput")
tposition = baseType("tposition")

class Grid():
    """
       Represents a grid.
    """
    def __init__(self, grid):
        self.grid = np.array(grid)
        self.pos = (0, 0)

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
            grid=np.array(grid)

            x_range, y_range = np.nonzero(grid)

            # for black shapes
            if len(x_range) == 0 and len(y_range) == 0:
                return (0, 0), grid

            pos = min(x_range), min(y_range)
            # cut= grid[min(y_range):max(y_range)+1][min(x_range):max(x_range)+1]
            cut= grid[min(x_range):max(x_range)+1,min(y_range):max(y_range)+1]
            return pos, cut

        pos2, cut = cutout(grid)
        if pos is None: pos = pos2
        super().__init__(cut)
        self.pos = pos
        self.index = index
        # self.area = np.sum(grid>0)

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
    def __init__(self, input_grid, training_examples):
        self.input_grid = Grid(input_grid)
        # all the examples
        self.grids = [(Grid(ex["input"]), Grid(ex["output"])) for ex in
                training_examples]

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
        if i < 0 or i >= len(l):
            raise ValueError()
        return l[i]

    return lambda i: get(l, i)

def _get_first(l):
    return l[1]

def _get_last(l):
    return l[-1]

def _length(l):
    return len(l)

def _remove_head(l):
    return l[1:]

def _sortby(l):
    return lambda f: sorted(l, key=f)

def _map(l):
    return lambda f: [f(x) for x in l]

def _apply(l):
    return lambda arg: [f(arg) for f in l]

def _zip(list1): 
    return lambda list2: lambda f: list(map(lambda x,y: f(x)(y), list1, list2))

def _compare(f):
    return lambda a: lambda b: f(a)==f(b)


def _filter_list(l):
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

# def _y_split(g):
    # find row with all one color, return left and right
    # np.
    np.all(g.grid == g.grid[:, 0], axis = 0)


# grid primitives
def _absolute_grid(g):
    return Grid(g.absolute_grid())

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
    if 0 in colors: colors.remove(0) # don't want black!
    return colors

def _get_object(g): #TODO figure out what this needs
    return Object(g.grid, pos=(0,0))

def _pixel2(c):
    return Pixel(np.array([[c]]), pos=(0, 0))

def _pixel(g):
    return lambda i: lambda j: Pixel(g.grid[i:i+1,j:j+1], (i, j))

def _overlay(g):
    return lambda g2: _stack_no_crop([g, g2])

def _list_of(g):
    return lambda g2: [g, g2]

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

def _objects_by_color(g):
    l = [_filter_color(g)(color) for color in range(MAX_COLOR+1)]
    l = [_object(a) for a in l if np.sum(a.grid) != 0]
    return l
        
def _objects(g):
    """
    Returns list of objects in grid (not including black) sorted by position
    """
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
    # TODO: always have relative positions?
    pixel_grid = [[Pixel(g.grid[i:i+1, j:j+1], (i + g.pos[0], j + g.pos[1])) 
            for i in range(len(g.grid))]
            for j in range(len(g.grid[0]))]
    # flattens nested list into single list
    return [item for sublist in pixel_grid for item in sublist]

def _hollow_objects(g):
    def hollow_objects(g):
        m = np.copy(g.grid)
        entriesToChange = []
        for i in range(1, len(m)-1):
            for j in range(1, len(m[i])-1):
                if(m[i][j]==m[i-1][j] and m[i][j]==m[i+1][j] and m[i][j]==m[i][j-1] and m[i][j]==m[i][j+1]):
                    entriesToChange.append([i, j])
        for entry in entriesToChange:
            m[entry[0]][entry[1]] = 0
        return Grid(m)
    return hollow_objects(g)

def _fill_line(g):
    def fill_line(g, background_color, line_color, color_to_add):
         m = np.copy(g.grid)
         for i in range(0, len(m)):
            for j in range(1, len(m[i])-1):
                if(m[i][j-1] == line_color and m[i][j] == background_color and m[i][j+1] == line_color):
                    m[i][j] = color_to_add
         return Grid(m)
    return lambda background_color: fill_line(g, background_color, 1, 2)
    #return lambda background_color: lambda line_color: lambda color_to_add: fill_line(g, background_color, line_color, color_to_add)

def _y_mirror(g):
    return Grid(np.flip(g.grid, axis=1))

def _x_mirror(g):
    return Grid(np.flip(g.grid, axis=0))

def _clockwise_rotate(g):
    def clockwise_rotate(g):
        m = np.copy(g.grid)
        rotatedArray = np.empty([len(m), len(m[0])], dtype=int)
        for i in range(0, len(m)):
            for j in range(0, len(m[i])):
                rotatedArray[i][j] = m[len(m)-1-j][i]
        return Grid(rotatedArray)
    return clockwise_rotate(g)

def _combine_grids_horizontally(g1):
    def combine_grids_horizontally(g1, g2):
        m1 = np.copy(g1.grid)
        m2 = np.copy(g2.grid)
        m = np.column_stack([m1, m2])
        return Grid(m)
    return lambda g2: combine_grids_horizontally(g1, g2)
    
def _combine_grids_vertically(g1):
    def combine_grids_vertically(g1, g2):
        m1 = np.copy(g1.grid)
        m2 = np.copy(g2.grid)
        m = np.concatenate([m1, m2])
        return Grid(m)
    return lambda g2: combine_grids_vertically(g1, g2)

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

    return Grid(grid.grid.astype("int"))

def _stack_no_crop(l):
    # stacks based on positions atop each other, masking first to last
    # assumes the grids are all the same size
    stackedgrid = np.zeros(shape=l[0].grid.shape)
    for g in l:
        # mask later additions
        stackedgrid += g.grid * (stackedgrid == 0)

    return Grid(stackedgrid.astype("int"))



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

def _color_in_grid(g):
    def color_in_grid(g, c):
        grid = g.grid
        grid[grid != 0] = c
        return Grid(grid)

    return lambda c: color_in_grid(g, c)

def _flood_fill(g):
    def flood_fill(g, c):
        grid = np.ones(shape=g.grid.shape).astype("int")*c
        return Grid(grid)

    return lambda c: flood_fill(g, c)

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

def _top_half(g):
    return Grid(g.grid[0:int(len(g.grid)/2), :])

def _bottom_half(g):
    return Grid(g.grid[int(len(g.grid)/2):, :])

def _left_half(g):
    return Grid(g.grid[:, 0:int(len(g.grid[0])/2)])

def _right_half(g):
    return Grid(g.grid[:, int(len(g.grid[0])/2):])

def _has_y_symmetry(g):
    return np.array_equal(np.flip(g.grid, axis=1), g.grid)

def _has_x_symmetry(g):
    return np.array_equal(np.flip(g.grid, axis=0), g.grid)

def _has_color(o):
    return lambda c: o.color == c

def _has_rotational_symmetry(g):
    return np.array_equal(_clockwise_rotate(g).grid, g.grid)

def _draw_connecting_line(g):
    # takes in the grid, the starting object, and list of objects to connect to
    # draws each line on a separate grid, then returns the grid with the stack
    def draw_connecting_line(g, o1, l):
        grids = []
        for o2 in l:
            # start with empty grid
            gridx,gridy = g.grid.shape
            line = np.zeros(shape=(gridx,gridy)).astype("int")

            # draw line between two positions
            startx, starty = o1.pos
            endx, endy = o2.pos

            sign = lambda a: 1 if a>0 else -1 if a<0 else 0

            x_step = sign(endx-startx) # normalize it, just figure out if its 1,0,-1
            y_step = sign(endy-starty) # normalize it, just figure out if its 1,0,-1

            x,y=startx, starty
            try: # you might end up off the grid if the steps don't line up neatly
                while not (x==endx and y==endy):
                    x += x_step
                    y += y_step
                    line[x][y]=1
            except:
                pass

            grids.append(Grid(line))

        return _stack_no_crop(grids)

    return lambda b: lambda c: draw_connecting_line(g,b,c)

def _draw_line(g):

    def draw_line(g, o, d):

        gridx,gridy = g.grid.shape
        line = np.zeros(shape=(gridx,gridy)).astype("int")

        # dir can be 0 45 90 135 180 ... 315 (degrees)
        # but we convert to radians
        direction=radians(d)

        y,x=o.pos
        while x < gridx and x >= 0 and y < gridy and y >= 0:
            line[y][x]=1
            x,y=int(round(x+cos(direction))), int(round(y-sin(direction)))

        # go in both directions
        bothways=True
        if bothways:
            direction=radians(d+180)

            y,x=o.pos
            while x < gridx and x >= 0 and y < gridy and y >= 0:
                line[y][x]=1
                x,y=int(round(x+cos(direction))), int(round(y-sin(direction)))


        return Grid(line)

    return lambda o: lambda d: draw_line(g,o,d)


## making the actual primitives

colors = {
    'color'+str(i): Primitive("color"+str(i), tcolor, i) for i in range(0, MAX_COLOR + 1)
    }
directions = {
    'dir'+str(i): Primitive('dir'+str(i), tdir, i) for i in range(0, 360, 45)
    }

ints = {
    str(i): Primitive(str(i), tint, i) for i in range(0, MAX_INT + 1)
    }
bools = {
    "True": Primitive("True", tbool, True),
    "False": Primitive("False", tbool, False)
    }

list_primitives = {
    "get": Primitive("get", arrow(tlist(t0), tint, t0), _get),
    "get_first": Primitive("get_first", arrow(tlist(t0), t0), _get_first),
    "get_last": Primitive("get_last", arrow(tlist(t0), t0), _get_last),
    "length": Primitive("length", arrow(tlist(t0), tint), _length),
    "remove_head": Primitive("remove_head", arrow(tlist(t0), t0), _remove_head),
    "sortby": Primitive("sortby", arrow(tlist(t0), arrow(t0, t1), tlist(t0)), _sortby),
    "map": Primitive("map", arrow(tlist(t0), arrow(t0, t1), tlist(t1)), _map),
    "filter_list": Primitive("filter_list", arrow(tlist(t0), arrow(t0, tbool), tlist(t0)), _filter_list),
    "compare": Primitive("compare", arrow(arrow(t0, t1), t0, t0, tbool), _compare),    
    "zip": Primitive("zip", arrow(arrow(t0, t1), t0, t0, tbool), _zip),    
    "reverse": Primitive("reverse", arrow(tlist(t0), tlist(t0)), _reverse),
    "apply_colors": Primitive("apply_colors", arrow(tlist(tgrid), tlist(tcolor)), _apply_colors)
    }

line_primitives = {
    # "draw_line": Primitive("apply_colors", arrow(tlist(tgrid), tlist(tcolor)), _apply_colors)
    "draw_connecting_line": Primitive("draw_connecting_line", arrow(tgrid, tobject, tlist(tobject), tgrid), _draw_connecting_line),
    "draw_line": Primitive("draw_connecting_line", arrow(tgrid, tobject, tdir, tbool, tgrid), _draw_line)
}

grid_primitives = {
    "map_i_to_j": Primitive("map_i_to_j", arrow(tgrid, tcolor, tcolor, tgrid), _map_i_to_j),
    "absolute_grid": Primitive("absolute_grid", arrow(tgrid, tgrid), _absolute_grid),
    "find_in_list": Primitive("find_in_list", arrow(tlist(tgrid), tint), _find_in_list),
    "find_in_grid": Primitive("find_in_grid", arrow(tgrid, tgrid, tposition), _find_in_grid),
    "filter_color": Primitive("filter_color", arrow(tgrid, tcolor, tgrid), _filter_color),
    "colors": Primitive("colors", arrow(tgrid, tlist(tcolor)), _colors),
    "color_of_obj": Primitive("color", arrow(tobject, tcolor), _color),
    "color": Primitive("color", arrow(tgrid, tcolor), _color),
    "objects": Primitive("objects", arrow(tgrid, tlist(tgrid)), _objects),
    "objects_by_color": Primitive("objects_by_color", arrow(tgrid, tlist(tgrid)), _objects_by_color),
    "get_object": Primitive("get_object", arrow(tgrid, tgrid), _get_object),
    "pixel2": Primitive("pixel2", arrow(tcolor, tgrid), _pixel2),
    "pixel": Primitive("pixel", arrow(tint, tint, tgrid), _pixel),
    "overlay": Primitive("overlay", arrow(tgrid, tgrid, tgrid), _overlay),
    "list_of": Primitive("list_of", arrow(tgrid, tgrid, tlist(tgrid)), _list_of),
    "pixels": Primitive("pixels", arrow(tgrid, tlist(tgrid)), _pixels),
    "set_shape": Primitive("set_shape", arrow(tgrid, tposition, tgrid), _set_shape),
    "shape": Primitive("shape", arrow(tgrid, tposition), _shape),
    "y_mirror": Primitive("y_mirror", arrow(tgrid, tgrid), _y_mirror),
    "x_mirror": Primitive("x_mirror", arrow(tgrid, tgrid), _x_mirror),
    "clockwise_rotate": Primitive("clockwise_rotate", arrow(tgrid, tgrid), _clockwise_rotate),
    "has_x_symmetry": Primitive("has_x_symmetry", arrow(tgrid, tbool), _has_x_symmetry),
    "has_y_symmetry": Primitive("has_y_symmetry", arrow(tgrid, tbool), _has_y_symmetry),
    "has_rotational_symmetry": Primitive("has_rotational_symmetry", arrow(tgrid, tbool), _has_rotational_symmetry),
    }

input_primitives = {
    "input": Primitive("input", arrow(tinput, tgrid), _input),
    "inputs": Primitive("inputs", arrow(tinput, tlist(tgrid)), _inputs),
    "outputs": Primitive("outputs", arrow(tinput, tlist(tgrid)), _outputs),
    "find_corresponding": Primitive("find_corresponding", arrow(tinput, tgrid, tgrid), _find_corresponding)
    }

list_consolidation = {
    "vstack": Primitive("vstack", arrow(tlist(tgrid), tgrid), _vstack),
    "hstack": Primitive("hstack", arrow(tlist(tgrid), tgrid), _hstack),
    "positionless_stack": Primitive("positionless_stack", arrow(tlist(tgrid), tgrid), _positionless_stack),
    "stack": Primitive("stack", arrow(tlist(tgrid), tgrid), _stack),
    "stack_no_crop": Primitive("stack_no_crop", arrow(tlist(tgrid), tgrid), _stack_no_crop),
    "combine_grids_horizontally": Primitive("combine_grids_horizontally", arrow(tgrid, tgrid, tgrid), _combine_grids_horizontally),
    "combine_grids_vertically": Primitive("combine_grids_vertically", arrow(tgrid, tgrid, tgrid), _combine_grids_vertically),
    }

boolean_primitives = {
    "and": Primitive("and", arrow(tbool, tbool, tbool), _and),
    "or": Primitive("or", arrow(tbool, tbool, tbool), _or),
    "not": Primitive("not", arrow(tbool, tbool), _not),
    "ite": Primitive("ite", arrow(tbool, t0, t0, t0), _ite),
    "eq": Primitive("eq", arrow(t0, t0, tbool), _eq)
    }

object_primitives = {
    "index": Primitive("index", arrow(tgrid, tint), _index),
    "position": Primitive("position", arrow(tgrid, tposition), _position),
    "x": Primitive("x", arrow(tgrid, tint), _x),
    "y": Primitive("y", arrow(tgrid, tint), _y),
    "color_in": Primitive("color_in", arrow(tgrid, tcolor, tgrid), _color_in),
    "flood_fill": Primitive("flood_fill", arrow(tgrid, tcolor, tgrid), _flood_fill),
    "size": Primitive("size", arrow(tgrid, tint), _size),
    "area": Primitive("area", arrow(tgrid, tint), _area)
    }

misc_primitives = {
    "inflate": Primitive("inflate", arrow(tgrid, tgrid), _inflate),
    "top_half": Primitive("top_half", arrow(tgrid, tgrid), _top_half),
    "bottom_half": Primitive("bottom_half", arrow(tgrid, tgrid), _bottom_half),
    "left_half": Primitive("left_half", arrow(tgrid, tgrid), _left_half),
    "right_half": Primitive("right_half", arrow(tgrid, tgrid), _right_half),
    }

primitive_dict = {**colors, **directions, **ints, **bools, **list_primitives,
        **line_primitives,
        **grid_primitives, **input_primitives, **list_consolidation,
        **boolean_primitives, **object_primitives, **misc_primitives}

primitives = list(primitive_dict.values())
