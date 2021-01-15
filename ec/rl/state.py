from anytree import Node, RenderTree
import numpy as np

class State():
    """
    The state is (1) a tree corresponding to a given ARC task and (2) a dictionary

    The tree:
        Left is input is leaf
        Right is output is root

        So parent is synonmous with rightnode
        And child is synonmous with leftnode

    The dictionary:
        Maps nodes in the tree to the action that connects them
    """
    def __init__(self, start_grid, end_grid):
        self.start = Node(start_grid)
        self.end = Node(end_grid)

        self.nodes = [self.start, self.end]

        self.actions = {} # a dictionary keeping track of what action took one node to another

    def extend_left_side(self, leftnode, newrightobject, action):
        """ 
        Append a node from the left part of the tree (add a parent)
        newnode could an object of be any of our ARC types e.g. a grid, a number, color, etc.
        """
        newrightnode=Node(newrightobject)
        leftnode.parent=newrightnode

        # self.nodes.append(n)
        self.actions[[leftnode, newrightnode]] = action

    def extend_left_side(self, rightnode, newleftobject, action):
        """ 
        Append a node from the right part of the tree (add a child)
        newnode could an object of be any of our ARC types e.g. a grid, a number, color, etc.
        """
        newleftnode = Node(newleftobject, parent=rightnode)

        # self.nodes.append(n)
        self.actions[[newleftnode,rightnode]]  = action

    def done(self):
        # how do we know the state is done?
        # everytime we connect a part on the left to the part on the right
        # we check that the part on the right has all its siblings completed.  if so, go up the tree one more step
        # then go up one more, and check all siblings completed.
        # if you reach the root, you're done.
        pass

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
