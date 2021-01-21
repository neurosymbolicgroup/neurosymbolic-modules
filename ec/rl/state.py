from typing import Any, List, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from operations import Op, Function, ValueNode, ProgramNode



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

    def get_value_nodes(self):
        return [node for node in self.graph.nodes if isinstance(node, ValueNode)]

    def value_node_exists(self, valuenode):
        """
        Checks if a valuenode with that value already exists in the tree
        If it does, return it
        If it doesn't, return None
        """
        return next((x for x in self.graph.nodes if hash(x)== hash(valuenode)), None)

    def check_invariants(self):
        assert nx.algorithms.dag.is_directed_acyclic_graph(self.fgraph)

    # from state_interface import add_hyperedge#, update_groundedness
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

        # for node in self.get_value_nodes():
        #     print(node.is_grounded)


        # # but, even if they got auto-merged, we still need to make sure its is_grounded is updated
        # if(state.value_node_exists(out_node)):
        #     print("exists")
        #     print(out_node.is_grounded)


    # # def extend_forward(self, fn: Function, inputs: List[ValueNode]):
    # def extend_forward(self, fn, inputs):
    #     assert len(fn.arg_types) == len(inputs)
    #     p = ProgramNode(fn, in_values=inputs)
    #     for inp in inputs:
    #         self.graph.add_edge(inp,p,label=str(inp.value)) # the graph infers new nodes from a collection of edges

    # # def extend_backward(self, fn: InvertibleFunction, inputs: List[ValueNode]):
    # def extend_backward(self, fn, inputs):
    #     assert len(fn.arg_types) == len(inputs)
    #     p = ProgramNode(fn, out_values=inputs)
    #     for inp in inputs:
    #         self.graph.add_edge(p, inp)


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

    import sys; sys.path.append("..") # hack to make importing bidir work
    from bidir.primitives.functions import rotate_ccw, rotate_cw
    from bidir.primitives.types import Grid

    from operations import take_forward_op

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
    take_forward_op(state, op, [state.start])   
    state.draw()

def arcexample_backward():

    import sys; sys.path.append("..") # hack to make importing bidir work
    from bidir.primitives.functions import rotate_ccw, rotate_cw
    from bidir.primitives.types import Grid

    from operations import take_inverse_op

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
    op = Op(rotate_ccw_func, rotate_cw_func, 'backward')

    # extend in the forward direction using fn and tuple of arguments that fn takes
    take_inverse_op(state, op, state.end)   
    state.draw()






if __name__ == '__main__':

    # arcexample_forward()
    arcexample_backward()
