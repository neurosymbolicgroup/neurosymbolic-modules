from typing import Any, List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from operations import Op, Function


class ValueNode:
    """
    Value nodes are what we called "input ports" and "output ports".
    They have a value, and are either grounded or not grounded.

    Values are a list of objects that the function evaluates to at that point
    (one object for each training example)

    All the value nodes are contained inside a program node
    (they are the input and output values to a that program node)

    All the actual edges drawn in the graph are between ValueNodes

    Any node that comes from the left side (from the input) should always be grounded).  
    Nodes that come from the right side are not grounded, until ALL of their inputs are grounded. 
    """
    def __init__(self, value: Any, is_grounded):
        self.value = value
        self.is_grounded = is_grounded

    def __str__(self):
        """
        Make the string representation just the value of the first training
        example (This function would just be for debugging purposes)
        """
        grounded=""
        if self.is_grounded:
            grounded = "\nGrounded"
        return "Val:\n" + str(self.value[0]) + grounded

    def __hash__(self):
        return hash(tuple(self.value))

    def __eq__(self, other):
        return hash(self) == hash(other)


class ProgramNode:
    """
    We have NetworkX Nodes, ProgramNodes (which hold the functions), and
    ValueNodes (which hold objects) Each ProgramNode knows its associated
    in-ValueNodes and out-ValueNodes ValueNodes are what we used to call
    "ports".  So in_values are in_ports and out_values are out_ports if you
    collapse the ValueNodes into one ProgramNode, you end up with the hyperdag

    The start and end nodes are the only program nodes that don't have an
    associated function

    Any node that comes from the left side (from the input) should always be
    grounded).  Nodes that come from the right side are not grounded, until ALL
    of their inputs are grounded.
    """
    def __init__(self, fn, in_values=[], out_values=[]):
        # a ValueNode for each of its in_port values
        self.in_values = in_values
        # a ValueNode for each of its out_port values
        self.out_values = out_values
        self.fn = fn

        # is_grounded if all inputs are grounded
        if all([val.is_grounded for val in in_values]):
            self.is_grounded = True
        else:
            self.is_grounded = False

    def __str__(self):
        """
        Return the name of the function and a unique identifier (Need the
        identifier because networkx needs the string representations for each
        node to be unique)
        """
        grounded=""
        if self.is_grounded:
            grounded = "\nGrounded"
        return "Fn:\n" + str(self.fn) +grounded


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

        # the start node is grounded, the end node won't be until all its inputs are grounded
        self.start = ValueNode(start_grids, is_grounded=True)
        self.end = ValueNode(end_grids, is_grounded=False)

        self.graph.add_node(self.start)#ProgramNode(fn=None, in_values=[self.start]))
        self.graph.add_node(self.end)#ProgramNode(fn=None, in_values=[self.end]))

    def get_value_nodes(self) -> List[ValueNode]:
        return [node for node in self.graph.nodes if isinstance(node, ValueNode)]

    def value_node_exists(self, valuenode):
        """
        Checks if a valuenode with that value already exists in the state
        If it does, return it
        If it doesn't, return None
        """
        return next((x for x in self.graph.nodes if hash(x)== hash(valuenode)), None)

    def check_invariants(self):
        assert nx.algorithms.dag.is_directed_acyclic_graph(self.fgraph)

    def add_hyperedge(
        self,
        in_nodes: List[ValueNode],
        out_nodes: List[ValueNode],
        fn: Function
    ):
        """
        Adds the hyperedge to the data structure.
        This can be represented underneath however is most convenient.
        This method itself could even be changed, just go change where it's called

        Each ValueNode is really just an edge (should have one input, one output)
        Each ProgramNode is a true node
        """
        out_node = out_nodes[0]
        p = ProgramNode(fn, in_values=in_nodes, out_values=out_nodes)
        for in_node in in_nodes:
            # draw edge from value node to program node
            # then from program node to output node
            self.graph.add_edge(in_node,p) # the graph infers new nodes from a collection of edges
            self.graph.add_edge(p,out_nodes[0]) # the graph infers new nodes from a collection of edges

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


def arcexample_forward():
    """
    An example showing how the state updates 
    when we apply a single-argument function (rotate) in the forward direction
    """

    import sys; sys.path.append("..") # hack to make importing bidir work
    from bidir.primitives.functions import rotate_ccw, rotate_cw
    from bidir.primitives.types import Grid

    from actions import apply_forward_op

    start_grids = [
        Grid(np.array([[0, 0], [1, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]
    
    end_grids = [
        Grid(np.array([[0, 1], [0, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]
    state = State(start_grids, end_grids)


    state.draw()

    # create operation
    rotate_ccw_func = Function("rotateccw", rotate_ccw, [Grid], [Grid])
    rotate_cw_func = Function("rotatecw", rotate_cw, [Grid], [Grid])
    op = Op(rotate_ccw_func, rotate_cw_func, 'forward')

    # extend in the forward direction using fn and tuple of arguments that fn takes
    apply_forward_op(state, op, [state.start])   
    state.draw()


def arcexample_backward():
    """
    An example showing how the state updates 
    when we apply a single-argument function (rotate) in the backward direction
    """

    import sys; sys.path.append("..") # hack to make importing bidir work
    from bidir.primitives.functions import rotate_ccw, rotate_cw
    from bidir.primitives.types import Grid

    from actions import apply_inverse_op

    start_grids = [
        Grid(np.array([[0, 0], [1, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]
    
    end_grids = [
        Grid(np.array([[0, 1], [0, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]
    state = State(start_grids, end_grids)


    state.draw()

    # create operation
    rotate_ccw_func = Function("rotateccw", rotate_ccw, [Grid], [Grid])
    rotate_cw_func = Function("rotatecw", rotate_cw, [Grid], [Grid])
    op = Op(rotate_ccw_func, rotate_cw_func, 'inverse')

    # extend in the forward direction using fn and tuple of arguments that fn takes
    apply_inverse_op(state, op, state.end)
    state.draw()


def arcexample_multiarg_forward():
    """
    An example showing how the state updates 
    when we apply a multi-argument function (inflate) in the forward direction
    """

    import sys; sys.path.append("..") # hack to make importing bidir work
    from bidir.primitives.functions import inflate, deflate
    from bidir.primitives.types import Grid

    from actions import apply_forward_op

    start_grids = [
        Grid(np.array([[0, 0], [0, 0]])),
        Grid(np.array([[1, 1], [1, 1]]))
    ]
    
    end_grids = [
        Grid(np.array([[0, 1], [0, 1]])),
        Grid(np.array([[2, 2], [2, 2]]))
    ]
    state = State(start_grids, end_grids)


    state.draw()

    # create operation
    inflate_func = Function("inflate", inflate, [Grid, int], [Grid])
    deflate_func = Function("deflate", deflate, [Grid, int], [Grid])
    op = Op(inflate, deflate, 'forward')

    # extend in the forward direction using fn and tuple of arguments that fn takes
    apply_forward_op(state, op, [state.start])   
    state.draw()




if __name__ == '__main__':

    # arcexample_forward()
    arcexample_backward()
    # arcexample_multiarg_forward()
