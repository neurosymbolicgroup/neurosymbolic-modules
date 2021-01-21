from typing import Any, List, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from operations import Function, ValueNode

import sys; sys.path.append("..") # hack to make importing bidir work
from bidir.primitives.functions import rotate_ccw
from bidir.primitives.types import Grid



class ProgramNode:
    """
    We have NetworkX Nodes, ProgramNodes (which hold the functions), and ValueNodes (which hold objects)
    Each ProgramNode knows its associated in-ValueNodes and out-ValueNodes
    ValueNodes are what we used to call "ports".  So in_values are in_ports and out_values are out_ports
    if you collapse the ValueNodes into one ProgramNode, you end up with the hyperdag

    The start and end nodes are the only program nodes that don't have an associated function
    """
    def __init__(self, fn, in_values=[], out_values=[]):
        self.in_values = in_values # a ValueNode for each of its in_port values
        self.out_values = out_values # a ValueNode for each of its out_port values
        self.fn = fn

        # if this is on the left side, inports are on left side
        # if this is on right side, inports are on right side

        # is_grounded if all outports are grounded
        self.is_grounded = False 

    def __str__(self):
        """
        Return the name of the function and a unique identifier
        (Need the identifier because networkx needs the string representations for each node to be unique)
        """
        return "Fn:" + " " + str(self.fn) #+ " " + str(hash(self))[0:4]

class State():
    """
    The state is (1) a tree corresponding to a given ARC task and (2) a dictionary

    The tree:
        Left is input is leaf
        Right is output is root

    The dictionary:
        Maps nodes in the tree to the action that connects them
    """
    def __init__(self, start_grids, end_grids):
        """
        Initialize the DAG
        For more, see https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/
        """

        self.graph = nx.DiGraph()

        assert len(start_grids) == len(end_grids)
        self.num_grid_pairs = len(start_grids) # number of training examples

        # Forward graph. Should be a DAG.
        self.graph = nx.MultiDiGraph()

        # The start and end nodes are the only program nodes that don't have an associated function
        self.start = ValueNode(start_grids)
        self.end = ValueNode(end_grids)
        self.graph.add_node(self.start)#ProgramNode(fn=None, in_values=[self.start]))
        self.graph.add_node(self.end)#ProgramNode(fn=None, in_values=[self.end]))

    def check_invariants(self):
        assert nx.algorithms.dag.is_directed_acyclic_graph(self.fgraph)
        #

    # def extend_forward(self, fn: Function, inputs: List[ValueNode]):
    def extend_forward(self, fn, inputs):
        assert len(fn.arg_types) == len(inputs)
        p = ProgramNode(fn, in_values=inputs)
        for inp in inputs:
            self.graph.add_edge(inp,p,label=str(inp.value)) # the graph infers new nodes from a collection of edges

    # def extend_backward(self, fn: InvertibleFunction, inputs: List[ValueNode]):
    def extend_backward(self, fn, inputs):
        assert len(fn.arg_types) == len(inputs)
        p = ProgramNode(fn, out_values=inputs)
        for inp in inputs:
            self.graph.add_edge(p, inp)


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
        # nx.draw_networkx_edge_labels(self.graph,pos,edge_labels=edge_labels,font_color='red')
        plt.show()


def arcexample():


    start_grids = [np.array([[0, 0], [1, 1]])]
    end_grids = [np.array([[0, 1], [1, 0]])]
    state = State(start_grids, end_grids)

    # state.draw()

    rotate_func = Function("rotateccw", rotate_ccw, [Grid], [Grid])

    # extend in the forward direction using fn and tuple of arguments that fn takes
    state.extend_forward(rotate_func, (state.start,))

    state.draw()


if __name__ == '__main__':

    #strexample()
    arcexample()
