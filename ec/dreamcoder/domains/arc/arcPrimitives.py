from dreamcoder.program import *
from dreamcoder.domains.arc.arcInput import load_task

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


def _empty_grid(): return ArcState()

def _empty_grid2(): return lambda y: ArcState()


# primitive functions can only have one argument; I think this is a requirement
# of lamdba-calculus style functions. see towerPrimitives.py, listPrimitives.py
def _draw(x):
    return lambda y: lambda color: lambda state: state.draw(x, y, color)

    

tgrid = baseType("grid")

primitives = [
            # Primitive("draw", arrow(tint, tint, tint, tgrid, tgrid), _draw),
        # Primitive("empty", tgrid, _empty_grid)
        Primitive("empty2", arrow(tgrid, tgrid), _empty_grid2)
] + [Primitive(str(j), tint, j) for j in range(9)]

