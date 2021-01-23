from typing import List, Tuple
from bidir.primitives.types import Grid

from bidir.primitives.functions import Function
import matplotlib.pyplot as plt
import networkx as nx


class ValueNode:
    """
    Value nodes are what we called "input ports" and "output ports".  They have
    a value, and are either grounded or not grounded.

    Values are a list of objects that the function evaluates to at that point
    (one object for each training example)

    All the value nodes are contained inside a program node (they are the input
    and output values to a that program node)

    All the actual edges drawn in the graph are between ValueNodes

    Any node that comes from the left side (from the input) should always be
    grounded).  Nodes that come from the right side are not grounded, until ALL
    of their inputs are grounded.
    """
    def __init__(self, value: Tuple):
        # Tuple of training example values
        # for some reason mypy wasn't catching when
        # actions.take_forward_action instantiated with a list
        assert type(value) == tuple, f'got type {type(value)}'
        self.value = value
        self.is_grounded = False

    def __str__(self):
        """
        Make the string representation just the value of the first training
        example (This function would just be for debugging purposes)
        """
        grounded = ""
        if self.is_grounded:
            grounded = "(Grounded)"
        return "Val: " + grounded + "\n" + str(self.value[0])

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
    def __init__(
        self,
        fn: Function,
        in_values: Tuple[ValueNode, ...],
        out_value: ValueNode
    ):
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
        self.start = ValueNode(start_grids, is_grounded=True)
        self.end = ValueNode(end_grids, is_grounded=False)

        self.graph.add_node(self.start)
        self.graph.add_node(self.end)

    def get_or_make_value_node(self, value: Tuple, is_grounded: bool):
        """
        Checks if a value node with the given value already exists in the
        graph. If so, returns it. Otherwise, creates a new value node, and
        returns that.

        Grounding is done automatically by the addition of edges in
        add_hyperedge, so don't worry about grounding when creating!
        """
        node = ValueNode(value=value)

        existing_node = self.value_node_exists(node)
        if existing_node is not None:
            node = existing_node
            node.is_grounded = is_grounded

        return node

    def propagate_groundedness(
        self,
        newly_grounded_nodes: List[ValueNode],
    ) -> None:
        to_propagate_further = []
        for node in newly_grounded_nodes:
            print(f'node: {node}')
            for succ in self.graph.successors(node):
                print(f'succ: {succ}')
            for program_node in self.graph.successors(node):
                print(f'out_value: {program_node.out_value}')
                print(f'program_node: {program_node}')
                print(f'is_grounded: {program_node.out_value.is_grounded}')
                print(f'in_grounds: {[n.is_grounded for n in program_node.in_values]}')
                if (not program_node.out_value.is_grounded
                    and all([node.is_grounded
                             for node in program_node.in_values])):

                    print(f'New grounded node: {program_node.out_value}')
                    program_node.out_value.is_grounded = True
                    to_propagate_further.append(program_node.out_value)
        print(f'to_propagate_further: {to_propagate_further}')
        if len(to_propagate_further) > 0:
            self.propagate_groundedness(to_propagate_further)

    def get_value_nodes(self) -> List[ValueNode]:
        return [node for node in self.graph.nodes
                if isinstance(node, ValueNode)]

    def value_node_exists(self, valuenode):
        """
        Checks if a valuenode with that value already exists in the state.
        If it does, return it.
        If it doesn't, return None.
        """
        return next((x for x in self.graph.nodes
                     if hash(x) == hash(valuenode)), None)

    def check_invariants(self):
        assert nx.algorithms.dag.is_directed_acyclic_graph(self.graph)
        # TODO: make sure this accurately finds duplicates
        assert len(set(self.graph.nodes)) == len(self.graph.nodes), (
                'duplicate present')

    def add_hyperedge(
        self,
        in_nodes: Tuple[ValueNode, ...],
        out_node: ValueNode,
        fn: Function
    ):
        """
        Adds the hyperedge to the data structure.
        """
        p = ProgramNode(fn, in_values=in_nodes, out_value=out_node)
        for in_node in in_nodes:
            # draw edge from value node to program node
            # then from program node to output node
            # the graph infers new nodes from a collection of edges
            self.graph.add_edge(in_node, p)
            # the graph infers new nodes from a collection of edges
            self.graph.add_edge(p, out_node)

    def done(self):
        """
        Returns true if we've found a program that successfully solves the
        training examples for the task embedded in this graph.
        """
        # TODO: propogate groundedness
        return self.end.is_grounded

    def draw(self):
        pos = nx.random_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True)
        # edge_labels = nx.get_edge_attributes(self.graph,'label')
        # nx.draw_networkx_edge_labels(self.graph,pos,edge_labels=edge_labels,font_color='red')
        plt.show()
