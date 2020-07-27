from dreamcoder.program import *
from dreamcoder.domains.arc.arcInput import load_task
from dreamcoder.type import arrow, baseType, tint
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

from scipy.ndimage import measurements
import numpy as np

MAX_GRID_LENGTH = 30
MAX_COLOR = 9

tgrid = baseType("tgrid")
tobject = baseType("tobject")
t_arclist = baseType("t_arclist")
# this is for the ArcInput class to index into an example and its location. It's
# a tuple of (i, (x, y)); for example return the color at (x, y) in example i.
tlocation2 = baseType("tlocation2")
tinput = baseType("tinput")
tdict = baseType("tdict")


def _gridempty(a): return a.empty_grid()

def _map_i_to_j_python(i):
    return lambda j: lambda a: a.map_i_to_j(i, j)

def _transform3(a):
    return lambda c1: lambda c2: lambda c3: a.transform({1: c1, 2: c2, 3: c3})

def _transform4(a):
    return lambda c1: lambda c2: lambda c3: lambda c4: a.transform({1: c1, 2: c2, 3: c3, 8: c4})

def _transform5(a):
    return lambda c1: lambda c2: lambda c3: lambda c4: lambda c5: a.transform({1: c1, 2: c2, 3: c3, 8: c4, 9: c5})

def _transform6(a):
    return lambda c1: lambda c2: lambda c3: lambda c4: lambda c5: lambda c6: a.transform({1: c1, 2: c2, 3: c3, 4: c4, 5: c5, 8: c6})

def _transform7(a):
    return lambda c1: lambda c2: lambda c3: lambda c4: lambda c5: lambda c6: lambda c7: a.transform({1: c1, 2: c2, 3: c3, 4: c4, 5: c5, 6: c6, 8: c7})


def _get_objects(a): return a.get_objects()

def _filter(color): 
    return lambda a: a.filter(color)

def _apply_fn(l): return lambda f: l.apply_fn(f)

def _reverse_list(l): return l.reverse()

def _get(l): return lambda i: l.get(i)

def _ix(o): return o.ix()

def _color(o): return o.color

def _move_down(o): return o.move_down()
# def _getcolor(o): 
#     """
#     Gets the color of the given object

#     For example, if the object given is
#     0 5 0
#     5 5 5
#     0 5 0

#     Then it returns 5
#     """
#     return o.color

def _stack(l): return l.stack()

def _getobject(i):
    """
    Gets the ith object in the form of an array  

    For example, if the object is of color 5 and looks like a cross, returns
    0 5 0
    5 5 5
    0 5 0
    """
    return lambda a: a.get_object(i)

def _color_at_location2(l):
    return lambda location2: l.color_at_location2(location2)

def _location2_with_color(l):
    return lambda c: l.location2_with_color(c)

def _get_input_grid(i):
    return i.get_input_grid()

def _get_input_grids(i):
    return i.get_input_grids()

def _get_output_grids(i):
    return i.get_output_grids()

def _for_each_color(i):
    return lambda f: i.for_each_color(f)


map_primitives = [
    Primitive("mapitoj", arrow(tint, tint, tgrid, tgrid), _map_i_to_j_python),
]
grid_primitives = [
    # Primitive("transform3", arrow(*([tgrid] + [tint]*3 + [tgrid])), _transform3)
    # Primitive("transform4", arrow(*([tgrid] + [tint]*4 + [tgrid])), _transform4)
    # Primitive("transform5", arrow(*([tgrid] + [tint]*5 + [tgrid])), _transform5)
    # Primitive("transform6", arrow(*([tgrid] + [tint]*6 + [tgrid])), _transform6)
    # Primitive("transform7", arrow(*([tgrid] + [tint]*7 + [tgrid])), _transform7)

    # Primitive("gridempty", arrow(tgrid, tgrid), _gridempty),
    Primitive("get_objects", arrow(tgrid, t_arclist), _get_objects),
    Primitive("color", arrow(tobject, tint), _color),
    Primitive("get", arrow(t_arclist, tint, tobject), _get),
    Primitive("move_down", arrow(tobject, tgrid), _move_down)
    ]

list_primitives = [
    Primitive("apply_fn", arrow(t_arclist, arrow(tgrid, tgrid), t_arclist), _apply_fn),
    # Primitive("reverse", arrow(t_arclist, t_arclist), _reverse_list),
    Primitive("stack", arrow(t_arclist, tgrid), _stack),
    Primitive("color_at_location2", arrow(t_arclist, tlocation2, tint),
        _color_at_location2),
    Primitive("location2_with_color", arrow(t_arclist, tint, tlocation2),
        _location2_with_color),
]

input_primitives = [
    Primitive("get_input_grid", arrow(tinput, tgrid), _get_input_grid),
    Primitive("get_input_grids", arrow(tinput, t_arclist),
        _get_input_grids),
    Primitive("get_output_grids", arrow(tinput, t_arclist),
        _get_output_grids),
    Primitive("for_each_color", arrow(tinput, arrow(tgrid, arrow(tint, tgrid)),
        tgrid), _for_each_color)
    ]

color_primitives = [Primitive(str(i), tint, i) for i in range(0, MAX_COLOR + 1)]

primitives = map_primitives + grid_primitives+ list_primitives + input_primitives + color_primitives


class ArcList:
    def __init__(self, grids):
        self.grids = grids

    def stack(self):
        grid = np.zeros(self.grids[0].grid.shape)
        for g in self.grids:
            grid += g.grid

        return ArcExample(grid)

    def filter(self, f):
        return ArcList([g for g in self.grids if f(g)])

    def reverse(self):
        return ArcList(self.grids[::-1])

    def get(self, i):
        if i < 0 or i >= len(self.grids):
            raise ValueError()

        return self.grids[i]

    def apply_fn(self, f):
        return ArcList([f(g) for g in self.grids])

    def color_at_location2(self, location2):
        return self.grids[location2[0]].color_at_location(location2[1])

    def location2_with_color(self, color):
        for i in range(len(self.grids)):
            if self.grids[i].contains_color(color):
                r = np.where(self.grids[i].grid == color)
                loc = list(zip(r[0], r[1]))[0]
                return i, loc

        raise ValueError()

    def __str__(self):
        return str(self.grids)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.grids == other.grids
        # thought this would make it so we don't have to stack, but it doesn't
        # elif isinstance(other, ArcExample):
            # return self.stack() == other
        else:
            return False


class ArcObject:
    def __init__(self, grid):
        # a numpy array
        self.grid = grid
        self.color = self.get_color()

    def get_color(self):
        return np.amax(self.grid)

    def move_down(self):
        self.grid = np.roll(self.grid, 1, axis=0)
        return ArcExample(self.grid)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equal(self.grid, other.grid)
        else:
            return False

class ArcExample:
    '''
        Just a wrapper around the list type we're working with.
    '''
    def __init__(self, grid):
        # a numpy array
        if np.max(grid) > MAX_COLOR:
            raise ValueError()
        self.grid = np.array(grid)
        self.position = (0, 0)

    def color_at_location(self, location):
        try:
            return self.grid[location]
        except IndexError:
            raise ValueError()


    def empty_grid(self):
        return ArcExample(np.zeros(np.array(self.grid).shape).astype(int))

    def map_i_to_j(self, i, j):
        m = np.copy(self.grid)
        m[m==i] = j
        return ArcExample(m)

    def get_objects(self):
        m = np.copy(self.grid)

        # first get all objects
        objects = []

        for color_x in np.unique(m):
            # if different colors...then different objects
            # return an array with 1s where that color is, 0 elsewhere
            data_with_only_color_x = np.where(m==color_x, 1, 0) 
            #if items of the same color are separated...then different objects
            data_with_only_color_x_and_object_labels, num_features = measurements.label(data_with_only_color_x)
            for object_i in range(1,num_features+1):
                # return an array with the appropriate color where that object is, 0 elsewhere
                data_with_only_object_i = np.where(data_with_only_color_x_and_object_labels==object_i, color_x, 0) 
                objects.append(ArcObject(data_with_only_object_i))

        # return an ArcList of ArcObjects
        return ArcList(objects)


    def filter(self, color):
        m = np.copy(self.grid)
        m[m != color] = 0
        return ArcExample(m)

    def contains_color(self, color):
        return np.any(self.grid == color)

    def color(self):
        max_color = 0
        m = 0
        for color in range(1, MAX_COLOR + 1):
            m2 = np.sum(self.grid == color)
            if m2 > m:
                m = m2
                max_color = color

        return max_color

    def transform(self, color_map):
        m = np.copy(self.grid)
        for k, v in color_map.items():
            # look from input grid, so that you don't map twice
            m[self.grid == k] = v
        return ArcExample(m)

    # def get_objects(self):
    #     grids = []
    #     for color in range(1, MAX_COLOR + 1):
    #         if color not in self.grid:
    #             continue
    #         a = self.filter(color)
    #         grids.append(a)
    #     grids = sorted(grids, key=lambda g: np.sum(g.grid != 0))
    #     for i, g in enumerate(grids):
    #         g.index = i
        
    #     if len(grids) == 0:
    #         assert np.all(self.grid == 0), 'self.grid: {}'.format(self.grid)
    #         grids.append(ArcExample(self.grid))

    #     return ArcList(grids)

    def ix(self):
        if hasattr(self, "index"):
            return self.index
        else:
            raise ValueError()

    def __str__(self):
        return str(self.grid)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return np.array_equal(self.grid, other.grid)
        else:
            return False


class ArcInput:
    """
    This puts all of the inputs into one thing, instead of one per example, so
    that we can synthesize a solution that looks at different examples at once
    (like for Andy's solution to the map task).
    """

    def __init__(self, input_grid, training_dict):
        self.input_grid = ArcExample(input_grid)
        # all the examples
        self.grids = [(ArcExample(ex["input"]), ArcExample(ex["output"])) for ex in
                training_dict]

    def get_input_grid(self):
        return self.input_grid

    def get_input_grids(self):
        return ArcList([g[0] for g in self.grids])

    def get_output_grids(self):
        return ArcList([g[1] for g in self.grids])

    def for_each_color(self, f):
        # for each color found in input grid, apply f_color to the input grid,
        # then return it

        new_grid = self.input_grid

        for color in np.unique(self.input_grid.grid):
            a = self.get_input_grids().location2_with_color(color)
            if a is not None:
                new_grid = f(new_grid)(color)

        return new_grid

    def __str__(self):
        return "i: {}, grids={}".format(self.input_grid, self.grids)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.input_grid == other.input_grid and self.grids == other.grids
        else:
            return False

                





