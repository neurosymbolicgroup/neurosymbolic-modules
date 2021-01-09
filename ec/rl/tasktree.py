from anytree import Node, RenderTree
import numpy as np

class TaskTree():
    """
    Tree corresponding to a given ARC task
    The leaf is the input
    The root is the output
    """
    def __init__(self, start_grid, end_grid):
        self.start = Node(start_grid)
        self.end = Node(end_grid)

    def print(self):
        for pre, fill, node in RenderTree(self.end):
            print("%s%s" % (pre, node.name))

def strexample():
    udo = Node("Udo")
    marc = Node("Marc", parent=udo)
    lian = Node("Lian", parent=marc)
    dan = Node("Dan", parent=udo)
    jet = Node("Jet", parent=dan)
    jan = Node("Jan", parent=dan)
    joe = Node("Joe", parent=dan)


    for pre, fill, node in RenderTree(udo):
        print("%s%s" % (pre, node.name))
    # Udo
    # ├── Marc
    # │   └── Lian
    # └── Dan
    #     ├── Jet
    #     ├── Jan
    #     └── Joe

    # print(udo)
    # print(joe)
    # print(dan.children)

def arcexample():
    start = np.array([[0, 0], [1, 1]])
    end = np.array([[1, 1], [0, 0]])
    tasktree = TaskTree(start, end)

if __name__ == '__main__':
    arcexample()
