from dreamcoder.program import *
from dreamcoder.domains.arc.arcInput import load_task
from dreamcoder.type import arrow, baseType, tint
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

import numpy as np

MAX_GRID_LENGTH = 30
MAX_COLOR = 9

tgrid = baseType("tgrid")

def _gridempty(a): return a.empty_grid()

def _map_i_to_j_python(i):
    return lambda j: lambda a: a.map_i_to_j(i, j)

def _get_objects(a): return a.get_objects()

def _filter(color): 
    return lambda a: a.filter(color)

def _transform2(a):
    return lambda c1: lambda c2: lambda c3: lambda c4: lambda c5: lambda c6: lambda c7: a.transform({1: c1, 2: c2, 3: c3, 4: c4, 5: c5, 6: c6, 7: c7})

def _transform(a):
    return lambda c1: lambda c2: lambda c3: lambda c4: a.transform({1: c1, 2: c2, 3: c3, 8: c4})

def _getobject(i):
    """
    Gets the ith object in the form of an array  

    For example, if the object is of color 5 and looks like a cross, returns
    0 5 0
    5 5 5
    0 5 0
    """
    # first get all objects
    objects = []

    for color_x in np.unique(self.data):
        # if different colors...then different objects
        # return an array with 1s where that color is, 0 elsewhere
        data_with_only_color_x = np.where(self.data==color_x, 1, 0) 
        #if items of the same color are separated...then different objects
        data_with_only_color_x_and_object_labels, num_features = measurements.label(data_with_only_color_x)
        for object_i in range(1,num_features+1):
            # return an array with the appropriate color where that object is, 0 elsewhere
            data_with_only_object_i = np.where(data_with_only_color_x_and_object_labels==object_i, color_x, 0) 
            objects.append(data_with_only_object_i)

    # then get ith one
    return objects[i]

def _getcolor(obj):
    return 1

primitives = [
    # Primitive("transform2", arrow(*([tgrid] + [tint]*7 + [tgrid])), _transform2)
    Primitive("transform", arrow(tgrid, tint, tint, tint, tint, tgrid), _transform),
    # Primitive("gridempty", arrow(tgrid, tgrid), _gridempty),
    # Primitive("mapitoj", arrow(tint, tint, tgrid, tgrid), _map_i_to_j_python),
    # Primitive("getobject", arrow(tint, tgrid, tgrid), _getobject),
    # Primitive("getcolor", arrow(tgrid, tint), _getcolor)    
]  + [Primitive(str(i), tint, i) for i in range(0, MAX_COLOR + 1)]

class ArcList:
    def __init__(self, grids):
        self.grids = grids

    def stack(self):
        grid = np.zeros(30, 30)
        for g in self.grids:
            grid[:len(g), :len(g[0])] = g

        return ArcExample(grid)

    def __str__(self):
        return str(self.grids)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.grids == other.grids
        else:
            return False

class ArcExample:
    '''
        Just a wrapper around the list type we're working with.
    '''
    def __init__(self, grid):
        # a numpy array
        self.grid = grid
        self.position = (0, 0)

    def empty_grid(self):
        return ArcExample(np.zeros(np.array(self.grid).shape).astype(int))

    def map_i_to_j(self, i, j):
        m = np.copy(self.grid)
        m[m==i] = j
        return ArcExample(m)

    def filter(self, color):
        m = np.copy(self.grid)
        m[m != color] = 0
        return ArcExample(m)

    def transform(self, colorMap):
        m = np.copy(self.grid)
        for k, v in colorMap.items():
            m[m == k] = v
        return ArcExample(m)

    def get_objects(self):
        return ArcList([self.filter(color) for color in range(0, 10)])
        

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
    def __init__(self, info_dict):
        self.info_dict = info_dict
        self.train = info_dict['train']
        self.teset = info_dict['test']


    def num_examples(self): return len(self.train)

    def get_example(self, i): return self.train[i]

    def get_example_input(self, i): return self.get_example(i)['input']

    def get_example_output(self, i): return self.get_example(i)['output']


class ArcState:
    GRID_WIDTH = 30
    GRID_HEIGHT = 30
    NUM_COLORS = 10 # 0 through 9 are colors.

    def __init__(self):
        self.grid = [[0]*GRID_WIDTH for i in range(GRID_HEIGHT)]
        self.height = 30
        self.width = 30

    def __init__(self, grid):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])

    def __str__(self): return f"ArcS(b={self.grid},w={self.width},h={self.height})"
    def __repr__(self): return str(self)

    def draw(self, x, y, color):
        assert color >= 0 and color < NUM_COLORS, "invalid color: {}".format(color)
        assert x >= 0 and x <= self.width, "invalid x: {}".format(x)
        assert y >= 0 and y <= self.height, "invalid y: {}".format(y)

        new_grid = [grid[i][j] for i in range(len(grid[0])) for j in
                range(len(grid))]
        new_grid[y][x] = color
        return TowerState(new_grid)
