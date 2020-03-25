from copy import copy 
from arc import Grid
import numpy as np

class Program:
    def __init__(self, grid):
        self.grid = copy(grid)
        self.grid.title = "Output"

        self.a = [] # an array to store any information necessary during course of program
        
        self.transformations = [] # the steps of the program

    def get_sorted_objects(self):
        """
        Sort objects by decreasing size, and store them in the progra's array a
        """
        self.a = sorted(self.grid.objects, key=lambda x: x.size, reverse=True)

    def reset_grid(self, width, height):
        """
        Reset the grid to a blank grid of size width x height
        """
        self.grid = Grid(title=self.grid.title, data=np.zeros((height, width)))

    def draw_vertical_line(self, size, color, pos):
        """
        Draw vertical line of size and color
        Using pos=(row, col) as topmost position
        """
        row, col = pos
        self.grid.data[row:row+size, col] = color

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
            if args != (): transformation(*args)

        # return the copy
        return self.grid