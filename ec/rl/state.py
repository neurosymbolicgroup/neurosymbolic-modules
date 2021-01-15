from anytree import Node, RenderTree
import numpy as np

class State():
    """
    The state is (1) a tree corresponding to a given ARC task and (2) a dictionary

    The tree:
        The leaf is the input
        The root is the output

    The dictionary:
        Maps nodes in the tree to the action that connects them
    """
    def __init__(self, start_grid, end_grid):
        self.start = Node(start_grid)
        self.end = Node(end_grid)

        self.nodes = [self.start, self.end]

        self.actions = {} # a dictionary keeping track of what action took one node to another

    def add_node_to_start(self, parentnode, newobject, action):
        """ 
        append a node from the 'start' part of the tree 


        newobject could an object of be any of our ARC types e.g. a grid, a number, color, etc.
        """
        n = Node(newobject, parent=parentnode)
        self.nodes.append(n)

        self.actions[[parentnode, n]] = action

    def add_node_to_end(self, childnode, newobject, action):
        """ append a node from the 'end' part of the tree """
        n = Node(newobject)
        childnode.parent=n
        self.nodes.append(n)


        self.actions[[n,childnode]]  = action

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
    # output should be:
    # Udo
    # ├── Marc
    # │   └── Lian
    # └── Dan
    #     ├── Jet
    #     ├── Jan
    #     └── Joe

    # can also say things like
    #   print(dan.children)
    #   print(udo)

def arcexample():
    start = np.array([[0, 0], [1, 1]])
    end = np.array([[1, 1], [0, 0]])
    state = State(start, end)

    state.print()

if __name__ == '__main__':
    #strexample()
    arcexample()
