import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class State():
    """
    The state is (1) a tree corresponding to a given ARC task and (2) a dictionary

    The tree:
        Left is input is leaf
        Right is output is root

    The dictionary:
        Maps nodes in the tree to the action that connects them
    """
    def __init__(self, start_grid, end_grid):
        """
        Initialize the DAG
        For more, see https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/
        """

        self.graph = nx.DiGraph()

        self.start = self.to_tuple(start_grid)
        self.end = self.to_tuple(end_grid)

        self.graph.add_node(self.start)
        self.graph.add_node(self.end)

        self.actions = {} # a dictionary keeping track of what action took one node to another

    def extend_left_side(self, leftnode, newrightobject, action):
        """
        Append a node from the left part of the tree
        newnode could an object of be any of our ARC types e.g. a grid, a number, color, etc.
        """
        if isinstance(newrightobject, np.ndarray) or isinstance(newrightobject, list):
            newrightobject = self.to_tuple(newrightobject)

        self.graph.add_edge(leftnode,newrightobject,label=action) # the graph infers new nodes from a collection of edges

    def extend_right_side(self, newleftobject, rightnode, action):
        """
        Append a node from the right part of the tree (add a child)
        newnode could an object of be any of our ARC types e.g. a grid, a number, color, etc.
        """
        if isinstance(newleftobject, np.ndarray) or isinstance(newleftobject, list):
            newleftobject = self.to_tuple(newleftobject)

        self.graph.add_edge(newleftobject,rightnode,label=action) # the graph infers new nodes from a collection of edges

    def to_tuple(self, array):
        """
        turn array into tuple of tuples
        since lists and arrays aren't hashable, and therefore not eligible as nodes
        """
        # if its a list, turn into array
        if isinstance(array, list):
            array = np.array(array)

        # figure out degree of array
        degree = len(array.shape)

        # work accordingly
        if degree==1:
            return tuple(array)
        elif degree==2:
            return tuple(map(tuple, array))
        else:
            return Exception("mapping higher dimensional arrays to tuples hasn't been implemented yet.  see state.py")

    def done(self):
        # how do we know the state is done?
        # everytime we connect a part on the left to the part on the right
        # we check that the part on the right has all its siblings completed.  if so, go up the tree one more step
        # then go up one more, and check all siblings completed.
        # if you reach the root, you're done.
        pass

    def draw(self):
        pos = nx.random_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        edge_labels = nx.get_edge_attributes(self.graph,'label')
        nx.draw_networkx_edge_labels(self.graph,pos,edge_labels=edge_labels,font_color='red')
        plt.show()


def arcexample():
    start = np.array([[0, 0], [0, 0]])
    end = np.array([[9, 9], [9, 9]])
    state = State(start, end)

    state.extend_left_side(state.start,[1], "get1array")
    state.extend_right_side([8], state.end,"get8array")

    state.draw()


if __name__ == '__main__':
    #strexample()
    arcexample()
