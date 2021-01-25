from typing import Tuple, List, Union, Optional
from bidir.primitives.types import Grid
from rl.program import Program, ProgFunction, ProgConstant, ProgInputGrids
import networkx as nx


class ForwardNode:
    def __init__(self,
                 train_values: Tuple,
                 test_values: Tuple,
                 is_constant=False):
        self.train_values = train_values
        self.test_values = test_values
        self.is_constant = is_constant
        self.is_grounded = True

    def matches_inverse_node(self, inv_node: 'InverseNode') -> bool:
        train_good = self.train_values == inv_node.train_values
        test_good = self.test_values not in inv_node.test_negative_values
        return train_good and test_good

    def __str__(self):
        return str(self.train_values[0])

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.train_values, self.test_values))

    def __eq__(self, other):
        if not isinstance(other, ForwardNode):
            return False
        train_good = self.train_values == other.train_values
        test_good = self.test_values == other.test_values
        return train_good and test_good


class InverseNode:
    def __init__(
        self,
        train_values: Tuple,
        test_negative_values: List[Tuple] = [],
    ):
        self.train_values = train_values
        self.test_negative_values = test_negative_values
        self.test_values: Optional[List[Tuple]] = None
        self.matching_forward_node: Optional[ForwardNode] = None

    @property
    def is_grounded(self) -> bool:
        return self.test_values is not None

    def __str__(self):
        s = ""
        if self.is_grounded:
            s += "(Grounded)"
        s += str(self.train_values[0])
        if len(self.test_negative_values) > 0:
            s += f"\n{len(self.test_negative_values)} negative values"
        return s

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash((self.train_values, tuple(self.test_negative_values)))

    def __eq__(self, other):
        """
        We should always keep separately instantiated InverseNodes separate
        from each other. You never know whether there will be a negative test
        value added which differs between the two.
        """
        return self is other


ValueNode = Union[ForwardNode, InverseNode, None]


class ProgramNode:
    def __init__(self, op: 'Op', in_nodes: Tuple[ValueNode, ...],
                 out_node: ValueNode):
        self.fn = op.fn
        self.op = op
        self.in_nodes = in_nodes
        self.out_node = out_node

    def inputs_grounded(self) -> bool:
        """
        Note: just because the program node's out_node is grounded doesn't mean
        the in_nodes are grounded. The opposite could arise if the out_node is
        simultaneously the output of a different program node whose inputs are
        grounded.
        """
        return all(in_node.is_grounded for in_node in self.in_nodes)

    def __str__(self):
        """
        Return the name of the function and a unique identifier (Need the
        identifier because networkx needs the string representations for each
        node to be unique)
        """
        return f"Fn: {self.fn.name}, Op: {self.op.name}"

    def __repr__(self):
        return str(self)


class State:
    def __init__(
        self,
        train_pairs: Tuple[Tuple[Grid, Grid], ...],
        test_inputs: Tuple[Grid, ...],
    ):
        """
        Initialize the DAG
        For more info on how the underlying graph works, see
        https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/
        """

        self.num_train = len(train_pairs)
        self.num_test = len(test_inputs)
        self.graph = nx.MultiDiGraph()
        train_inputs, train_outputs = zip(*train_pairs)
        self.start = ForwardNode(train_inputs, test_inputs)
        self.end = InverseNode(train_outputs)

        self.graph.add_node(self.start)
        self.graph.add_node(self.end)

    def get_forward_nodes(self) -> List[ForwardNode]:
        return [
            node for node in self.graph.nodes if isinstance(node, ForwardNode)
        ]

    def get_inverse_nodes(self) -> List[InverseNode]:
        return [
            node for node in self.graph.nodes if isinstance(node, InverseNode)
        ]

    def get_value_nodes(self) -> List[ValueNode]:
        return [
            node for node in self.graph.nodes
            if isinstance(node, InverseNode) or isinstance(node, ForwardNode)
        ]

    def get_program_nodes(self) -> List[ProgramNode]:
        return [
            node for node in self.graph.nodes if isinstance(node, ProgramNode)
        ]

    def propagate_grounding(self, inv_node: InverseNode):
        assert inv_node.is_grounded

        for program_node in self.graph.successors(inv_node):
            assert isinstance(program_node, ProgramNode)
            in_nodes = program_node.in_nodes
            out_node = program_node.out_node
            if (not out_node.is_grounded
                    and all(node.is_grounded for node in in_nodes)):
                # shape (num_inputs, num_test)
                test_args: List[Tuple] = [
                    node.test_values for node in in_nodes
                ]

                def ith_args(test_args: Tuple[Tuple, ...], i: int) -> List:
                    return [arg[i] for arg in test_args]

                out_test_values = tuple(
                    program_node.fn.fn(*ith_args(test_args, i))
                    for i in range(self.num_test))

                out_node.test_values = out_test_values
                self.propagate_grounding(out_node)

    def process_forward_node(self, fw_node: ForwardNode):
        """
        Updates groundedness of any inverse nodes based on addition of this
        forward node.
        """
        for inv_node in self.get_inverse_nodes():
            self.possibly_match_nodes(fw_node=fw_node, inv_node=inv_node)

    def process_inverse_node(self, inv_node: InverseNode):
        for fw_node in self.get_forward_nodes():
            self.possibly_match_nodes(fw_node=fw_node, inv_node=inv_node)

    def possibly_match_nodes(self, fw_node: ForwardNode, inv_node: InverseNode):
        if (not inv_node.is_grounded
                and fw_node.matches_inverse_node(inv_node)):
            inv_node.test_values = fw_node.test_values
            # TODO may not need this edge?
            self.graph.add_edge(fw_node, inv_node, label='match')
            inv_node.matching_forward_node = fw_node
            self.propagate_grounding(inv_node)

    def add_hyperedge(self, in_nodes: Tuple[ValueNode, ...],
                      out_node: ValueNode, op: 'Op'):
        """
        Adds the hyperedge to the data structure.
        Also checks if any new nodes were added, and processes them if so
        (i.e. checking for grounding/matching.)
        """
        p = ProgramNode(op, in_nodes=in_nodes, out_node=out_node)
        for in_node in in_nodes:
            # if node wasn't present previously, this adds it to the graph
            self.graph.add_edge(in_node, p)

        self.graph.add_edge(p, out_node)


    def add_constant_node(self, node: ForwardNode):
        self.graph.add_node(node)

    @property
    def done(self):
        """
        Returns true if we've found a program that successfully solves the
        training examples for the task embedded in this graph.
        """
        return self.end.is_grounded

    def get_program(self) -> Optional[Program]:
        """
        If there is a program that solves the task, returns it.
        If there are multiple, just returns one of them.
        If there are none, returns None.
        """
        if not self.done:
            return None

        def find_subprogram(node: ValueNode) -> Program:
            assert node.is_grounded

            if self.start == node:
                return ProgInputGrids(self.start.train_values)
            if isinstance(node, ForwardNode) and node.is_constant:
                return ProgConstant(node.train_values[0])

            if (isinstance(node, InverseNode)
                    and node.matching_forward_node is not None):
                return find_subprogram(node.matching_forward_node)

            valid_prog_nodes = [
                prog_node for prog_node in self.graph.predecessors(node)
                if prog_node.inputs_grounded()
            ]

            assert len(valid_prog_nodes) > 0, (
                "Hmm... you told us the end was grounded, but somehow we got",
                "to an ungrounded node here.")

            prog_node = valid_prog_nodes[0]
            subprograms = []
            for in_value in prog_node.in_nodes:
                subprogram = find_subprogram(in_value)
                subprograms.append(subprogram)

            return ProgFunction(prog_node.fn, subprograms)

        return find_subprogram(self.end)
