from typing import List, Tuple, Optional
from bidir.primitives.types import Grid

from bidir.primitives.functions import Function
import matplotlib.pyplot as plt
import networkx as nx
from rl.program import Program, ProgFunction, ProgConstant, ProgInputGrids


class ValueNode:
    """
    Value nodes are what we called "input ports" and "output ports".  They have
    a value, and are either grounded or not grounded.

    Values are a tuple of objects that the function evaluates to at that point
    (one object for each training example)

    All the value nodes are contained inside a program node (they are the input
    and output values to a that program node)

    All the actual edges drawn in the graph are between ValueNodes and
    ProgramNodes.

    Groundedness is stored as attributes of the networkx graph. So really this
    is just a wrapper class for a tuple of example values.

    Any node that comes from the left side (from the input) should always be
    grounded).  Nodes that come from the right side are not grounded, until ALL
    of their inputs are grounded.
    """
    def __init__(self, value: Tuple):
        self.value = value

    def __str__(self):
        return str(self.value[0])

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return hash(self) == hash(other)


class ProgramNode:
    """
    We have NetworkX Nodes, ProgramNodes (which hold the functions), and
    ValueNodes (which hold objects) Each ProgramNode knows its associated
    in-ValueNodes and out-ValueNodes ValueNodes are what we used to call
    "ports".  So in_values are in_ports and out_values are out_ports if you
    collapse the ValueNodes into one ProgramNode, you end up with the hyperdag

    Any node that comes from the left side (from the input) should always be
    grounded).  Nodes that come from the right side are not grounded, until ALL
    of their inputs are grounded.
    """
    def __init__(self, fn: Function, in_values: Tuple[ValueNode, ...],
                 out_value: ValueNode):
        # a ValueNode for each of its in_port values
        self.in_values = in_values
        # a ValueNode for its out_port value
        self.out_value = out_value
        self.fn = fn

    def __str__(self):
        """
        Return the name of the function and a unique identifier (Need the
        identifier because networkx needs the string representations for each
        node to be unique)
        """
        return "Fn: " + str(self.fn)

    def __repr__(self):
        return str(self)


class State():
    """
    Represents the functions applied with a graph.
    Each node is either a ProgramNode (function application) or a ValueNode
    (input or output of a function).
    """
    def __init__(self, start_grids: Tuple[Grid], end_grids: Tuple[Grid]):
        """
        Initialize the DAG
        For more, see
        https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/
        """

        assert len(start_grids) == len(end_grids)
        self.num_examples = len(start_grids)  # number of training examples

        # Forward graph. Should be a DAG.
        self.graph = nx.MultiDiGraph()

        # the start node is grounded.
        # the end node won't be grounded until all its inputs are grounded.
        self.start = ValueNode(start_grids)
        self.end = ValueNode(end_grids)

        self.graph.add_node(self.start)
        self.graph.add_node(self.end)
        self.ground_value_node(self.start)

    def mark_as_constant(self, value_node: ValueNode):
        """
        Marks node as a constant node.
        Doing so grounds it too.
        """
        self.graph.add_node(value_node, constant=True)
        self.ground_value_node(value_node)

    def is_grounded(self, value_node: ValueNode):
        return "grounded" in self.graph.nodes[value_node]

    def is_constant(self, value_node: ValueNode):
        return "constant" in self.graph.nodes[value_node]

    def ground_value_node(self, value_node: ValueNode):
        """
        Grounds the given value node.
        Also recursively propagates groundedness throughout the graph.
        To do so, we check whether grounding this node
        grounds any previously ungrounded value nodes--that is, output value
        nodes whose inputs were all grounded except for this one. If so, then
        we ground that node, and continue recursively.
        """
        self.graph.nodes[value_node]["grounded"] = True

        for program_node in self.graph.successors(value_node):
            if (not self.is_grounded(program_node.out_value) and all(
                [self.is_grounded(node) for node in program_node.in_values])):

                self.ground_value_node(program_node.out_value)

    def get_value_nodes(self) -> List[ValueNode]:
        return [
            node for node in self.graph.nodes if isinstance(node, ValueNode)
        ]

    def get_program_nodes(self) -> List[ProgramNode]:
        return [
            node for node in self.graph.nodes if isinstance(node, ProgramNode)
        ]

    def check_invariants(self):
        assert nx.algorithms.dag.is_directed_acyclic_graph(self.graph)
        for program_node in self.get_program_nodes():
            in_nodes_grounded = all(
                [self.is_grounded(node) for node in program_node.in_values])
            out_node_grounded = self.is_grounded(program_node.out_value)
            assert in_nodes_grounded == out_node_grounded

    def add_hyperedge(self, in_nodes: Tuple[ValueNode, ...],
                      out_node: ValueNode, fn: Function):
        """
        Adds the hyperedge to the data structure.

        We also take care of updating groundedness here.
        In general, it suffices to check, whenever a hyperedge is made, whether
        all of the inputs are grounded. If so, then the output value will be
        grounded. This is true whether we add an edge due to a forward
        operation, inverse operation, or conditional inverse operation.
        """
        # if nodes already exist with this value, then it will not be
        # overwritten (and by it, I mean its groundedness!)
        p = ProgramNode(fn, in_values=in_nodes, out_value=out_node)
        for in_node in in_nodes:
            # draw edge from value node to program node
            # then from program node to output node
            # the graph infers new nodes from a collection of edges
            self.graph.add_edge(in_node, p)
            # the graph infers new nodes from a collection of edges
            self.graph.add_edge(p, out_node)

        if all([self.is_grounded(in_node) for in_node in in_nodes]):
            self.ground_value_node(out_node)

    def done(self):
        """
        Returns true if we've found a program that successfully solves the
        training examples for the task embedded in this graph.
        """
        return self.is_grounded(self.end)

    def get_program(self) -> Optional[Program]:
        """
        If there is a program that solves the task, returns it.
        If there are multiple, just returns one of them.
        If there are none, returns None.
        """
        def inputs_grounded(program_node: ProgramNode) -> bool:
            return all([
                self.is_grounded(in_value)
                for in_value in program_node.in_values
            ])

        def find_subprogram(node: ValueNode) -> Optional[Program]:
            if self.start == node:
                return ProgInputGrids(self.start.value)
            if self.is_constant(node):
                return ProgConstant(node.value[0])

            valid_prog_nodes = [
                prog_node for prog_node in self.graph.predecessors(node)
                if inputs_grounded(prog_node)
            ]
            if len(valid_prog_nodes) == 0:
                return None

            prog_node = valid_prog_nodes[0]
            subprograms = []
            for in_value in prog_node.in_values:
                subprogram = find_subprogram(in_value)
                if subprogram is None:
                    return None
                else:
                    subprograms.append(subprogram)

            return ProgFunction(prog_node.fn, subprograms)

        return find_subprogram(self.end)

    def draw(self):
        pos = nx.random_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        # edge_labels = nx.get_edge_attributes(self.graph,'label')
        # nx.draw_networkx_edge_labels(self.graph,pos,edge_labels=edge_labels,font_color='red')
        plt.show()
