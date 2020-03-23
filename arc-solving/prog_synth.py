from copy import copy 

class Program:
    def __init__(self, grid):
        self.grid = copy(grid)
        self.grid.title = "Output"
        
        self.transformations = [] # the steps of the program

    def set_transformations(self, transformations):
        self.transformations = transformations

    def apply_transformations(self):
        """
        Applies all the transformations in this program to the input grid, and returns a new grid
        """
        # create a copy of the main input grid
        # grid = self.grid.get_copy()

        # apply the transformations
        for transformation, args in self.transformations:
            transformation(*args)

        # return the copy
        return self.grid