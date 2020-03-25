from arc import Problem, Grid
from prog_synth import Program
import matplotlib.pyplot as plt

COLORS = {
    'black' : 0, 
    'blue' : 1, 
    'red': 2, 
    'green': 3, 
    'yellow': 4, 
    'grey': 5, 
    'pink': 6, 
    'orange': 7, 
    'lightblue': 8,
    'maroon': 9
}

def show(in_grid, out_grid, target_grid):
    plt.subplot(1,3,1)
    in_grid.show()

    plt.subplot(1,3,2)
    target_grid.show()

    plt.subplot(1,3,3)
    out_grid.show()

    plt.show()

def colors_problem():
    # --------------------------
    # load in an ARC problem
    # ---------------------------
    problem = Problem("272f95fa")

    training_example = 0
    in_grid = problem.get_grid(training_example, "input")
    target_grid = problem.get_grid(training_example, "target")
    
    # ---------------------------------------------------------
    # create a program that should always output the right grid
    # ---------------------------------------------------------
    program = Program(grid=in_grid)
    program.set_transformations([ 
        (program.grid.change_color, (program.grid.get_object(1), COLORS["red"])),   # get object by index 0, and color it red
        (program.grid.change_color, (program.grid.get_object(3), COLORS["yellow"])), 
        (program.grid.change_color, (program.grid.get_object(4), COLORS["pink"])), 
        (program.grid.change_color, (program.grid.get_object(5), COLORS["green"])), 
        (program.grid.change_color, (program.grid.get_object(7), COLORS["blue"])), 
        ])
    out_grid = program.apply_transformations()

    # ---------------------------------------------------------
    # show all grids
    # ---------------------------------------------------------
    show(in_grid, out_grid, target_grid)

def counting_problem():
    # --------------------------
    # load in an ARC problem
    # ---------------------------
    problem = Problem("9af7a82c")

    training_example = 0
    in_grid = problem.get_grid(training_example, "input")
    target_grid = problem.get_grid(training_example, "target")
    
    # ---------------------------------------------------------
    # create a program that should always output the right grid
    # ---------------------------------------------------------
    program = Program(grid=in_grid)
    program.set_transformations([ 
        (program.get_sorted_objects(), ()), # sort objects by size, put in array a
        (program.reset_grid, (len(program.a), program.a[0].size)), # create grid of specified w x h
        # (program.grid.draw_vertical_line, (program.a[0].get_size(), program.a[0].get_color(), (0,0))) # draw those objects as vertical lines
        ])
    out_grid = program.apply_transformations()

    # print(program.a[0].get_size())

    # ---------------------------------------------------------
    # show all grids
    # ---------------------------------------------------------
    show(in_grid, out_grid, target_grid)


if __name__ == '__main__':
    # colors_problem()
    counting_problem()

 
