from dreamcoder.program import *
from dreamcoder.domains.arc.arcInput import load_task
from dreamcoder.type import arrow, baseType, tint
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive

import numpy as np


tgrid = baseType("tgrid")

def _gridempty(a): return a.empty_grid()

def _map3to4(l): return a.map_i_to_j(3, 4)

def _map1to5(l): return a.map_i_to_j(1, 5)

def _map2to6(lO): return a.map_i_to_j(2, 6)

def _map_i_to_j_python(i):
    return lambda j: lambda a: a.map_i_to_j(i, j)

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
    Primitive("gridempty", arrow(tgrid, tgrid), _gridempty),
    Primitive("mapitoj", arrow(tint, tint, tgrid, tgrid), _map_i_to_j_python),
    Primitive("getobject", arrow(tint, tgrid, tgrid), _getobject),
    Primitive("getcolor", arrow(tgrid, tint), _getcolor)    
]  + [Primitive(str(i), tint, i) for i in range(1, 7)]

class ArcExample:
    '''
        Just a wrapper around the list type we're working with.
    '''
    def __init__(self, input_list):
        # a python list
        self.input_list = input_list

    def empty_grid(self):
        return ArcExample(np.zeros(np.array(self.input_list).shape).astype(int).tolist())

    def map_i_to_j(self, i, j):
        m = np.copy(self.input_list)
        m[m==i] = j
        return ArcExample(m.tolist())

    def __str__(self):
        return str(self.input_list)

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.input_list == other.input_list
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
