from dreamcoder.program import *
from dreamcoder.domains.arc.arcInput import load_task
from dreamcoder.type import arrow, baseType, tint
import numpy as np


tgrid = baseType("tgrid")

def _gridempty(a): return a.empty_grid()

def _map3to4(l): return a.map_i_to_j(3, 4)

def _map1to5(l): return a.map_i_to_j(1, 5)

def _map2to6(lO): return a.map_i_to_j(2, 6)

def _map_i_to_j_python(i):
    return lambda j: lambda a: a.map_i_to_j(i, j)

primitives =  [
    Primitive("gridempty", arrow(tgrid, tgrid), _gridempty),
    Primitive("map3to4", arrow(tgrid, tgrid), _map3to4),
    Primitive("map1to5", arrow(tgrid, tgrid), _map1to5),
    Primitive("map2to6", arrow(tgrid, tgrid), _map2to6),
    Primitive("mapitoj", arrow(tint, tint, tgrid, tgrid), _map_i_to_j_python)
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
