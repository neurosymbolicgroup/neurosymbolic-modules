from typing import List, Optional, Tuple, Set, Sequence
import matplotlib.pyplot as plt
import networkx as nx

from bidir.primitives.functions import Function
from rl.program import Program, ProgFunction, ProgConstant, ProgInput
from bidir.task_utils import Task
from bidir.utils import SynthError, timing


class ValueNode:
    """
    Value nodes are what we called "input ports" and "output ports". They have
    a value, and are either grounded or not grounded.

    Values are a tuple of objects that the function evaluates to at that point
    (one object for each training example)

    All the value nodes are contained inside a program node (they are the input
    and output values to a that program node)

    All the actual edges drawn in the graph are between ValueNodes and
    ProgramNodes.

    Really this is just a wrapper class for a tuple of example values.

    ValueNodes are meant to be immutable.
    Thus don't compare ValueNode using "is".
    Use "==" instead.
    """
    def __init__(self, value: Tuple):
        self._value: Tuple = value

    @property
    def value(self) -> Tuple:  # read-only
        return self._value

    def __str__(self):
        # return f"V({self._value[0]})"
        return str(self._value[0])

    def __repr__(self):
        return repr(self._value[0])

    def __hash__(self):
        return hash(self._value)

    def __eq__(self, other):
        if not hasattr(other, "_value"):
            return False
        return self._value == other._value


class ProgramNode:
    """
    We have NetworkX Nodes, ProgramNodes (which hold the functions), and
    ValueNodes (which hold objects). Each ProgramNode knows its associated
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
        out_value: ValueNode,
        action_num: int,
    ):
        # a ValueNode for each of its in_port values
        self.in_values = in_values
        # a ValueNode for its out_port value
        self.out_value = out_value
        self.fn = fn
        self.action_num = action_num

    def __str__(self):
        """
        Return the name of the function and a unique identifier (Need the
        identifier because networkx needs the string representations for each
        node to be unique)
        """
        # TODO: this isn't unique though?
        return "Fn: " + str(self.fn)

    def __repr__(self):
        return str(self)


class ProgramSearchGraph():
    """
    Represents the functions applied with a graph.
    Each node is either a ProgramNode (function application) or a ValueNode
    (input or output of a function).
    """
    def __init__(
        self,
        task: Task,
        additional_nodes: Sequence[Tuple[ValueNode, bool]] = None,
    ):
        """
        Initialize the DAG

        For more info on the graph used underneath, see
        https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/
        """
        self.task = task  # never used currently, I think

        self.num_examples = len(task.target)
        assert all(len(i) == self.num_examples for i in task.inputs)

        # Forward graph. Should be a DAG.
        self.graph = nx.MultiDiGraph()

        # set of examples for each input
        self.inputs = tuple(ValueNode(i) for i in task.inputs)
        for node in self.inputs:
            self.graph.add_node(node)
            self.ground_value_node(node)

        self.end = ValueNode(task.target)
        self.graph.add_node(self.end)

        if additional_nodes:
            for (node, grounded) in additional_nodes:
                self.graph.add_node(node)
                if grounded:
                    self.ground_value_node(node)

    def get_value_nodes(self) -> List[ValueNode]:
        return [n for n in self.graph.nodes if isinstance(n, ValueNode)]

    def get_program_nodes(self) -> List[ProgramNode]:
        return [n for n in self.graph.nodes if isinstance(n, ProgramNode)]

    def is_constant(self, v: ValueNode) -> bool:
        """
        Returns whether a ValueNode v is a constant in self.graph.

        Constantness is stored as an attribute of the self.graph NetworkX graph.
        """
        if v not in self.graph.nodes:
            return False
        return self.graph.nodes[v].get("constant", False)

    def is_grounded(self, v: ValueNode) -> bool:
        """
        Returns whether a ValueNode v is grounded.

        Groundedness is defined recursively via the following conditions:
            1. All start nodes are grounded.
            2. Constant nodes are grounded.
            3. The output of a ProgramNode whose inputs are grounded are
               all grounded.

        Groundedness is stored as an attribute of the self.graph NetworkX graph.
        """
        if v not in self.graph.nodes:
            return False
        return self.graph.nodes[v].get("grounded", False)

    def inputs_grounded(self, p: ProgramNode) -> bool:
        return all(self.is_grounded(iv) for iv in p.in_values)

    def remove_ungrounded_node(self, v: ValueNode):
        assert not self.is_grounded(v)

        for p in list(self.graph.predecessors(v)):
            self.graph.remove_node(p)
            for in_node in p.in_values:
                if not self.is_grounded(in_node):
                    self.remove_ungrounded_node(in_node)

        self.graph.remove_node(v)

    def ground_value_node(self, v: ValueNode) -> None:
        """
        Grounds the given value node.

        Also recursively propagates groundedness throughout the graph.  To do
        so, we check whether grounding this node grounds any previously
        ungrounded value nodes -- that is, output value nodes whose inputs were
        all grounded except for this one. If so, then we ground that node, and
        continue recursively.

        If we ground a node which had multiple inverse ops pointing towards it,
        then we remove the as-yet-ungrounded ops and their predecessors, since
        they're no longer needed.
        """
        # Do nothing if v is already grounded
        if self.is_grounded(v):
            return

        # Set grounded attribute
        self.graph.nodes[v]["grounded"] = True

        # if there are any inverse ops coming off of this node which weren't
        # grounded yet, then get rid of them. Then recursively get rid of any
        # of their predecessors
        for p in list(self.graph.predecessors(v)):
            if not self.inputs_grounded(p):
                self.graph.remove_node(p)
                for in_node in p.in_values:
                    if not self.is_grounded(in_node):
                        self.remove_ungrounded_node(in_node)

        # Recursively ground successors
        for p in self.graph.successors(v):
            if self.inputs_grounded(p):
                self.ground_value_node(p.out_value)

    def check_invariants(self):

        # Check start and end
        assert all(isinstance(i, ValueNode) for i in self.inputs)
        assert isinstance(self.end, ValueNode)

        # Check no edges between nodes of same type
        for (u, v) in self.graph.edges():
            assert any([
                isinstance(u, ValueNode) and isinstance(v, ProgramNode),
                isinstance(u, ProgramNode) and isinstance(v, ValueNode),
            ])

        # Check graph edges consistent with ProgramNode data
        for p in self.get_program_nodes():
            assert set(p.in_values) == set(self.graph.predecessors(p))
            assert set([p.out_value]) == set(self.graph.successors(p))

        # Check graph is acyclic
        assert nx.algorithms.dag.is_directed_acyclic_graph(self.graph)

        # Check groundedness
        for p in self.get_program_nodes():
            assert self.inputs_grounded(p) == self.is_grounded(p.out_value), (
                f"in: {p.in_values}, out:{p.out_value}, fn: {p.fn.name}")

        # Check for duplicate program nodes
        assert len(self.get_program_nodes()) == len({
            (frozenset(p.in_values), p.out_value)
            for p in self.get_program_nodes()
        })

    def add_constant(self, value_node: ValueNode, action_num: int) -> None:
        """
        Adds v as a constant and grounded node.
        """
        self.graph.add_node(value_node, constant=True, action_num=action_num)
        self.ground_value_node(value_node)

    def add_hyperedge(
        self,
        in_nodes: Tuple[ValueNode, ...],
        out_node: ValueNode,
        fn: Function,
        action_num: int,
    ):
        """
        Adds the hyperedge to the data structure.

        We also take care of updating groundedness here.
        In general, it suffices to check, whenever a hyperedge is made, whether
        all of the inputs are grounded. If so, then the output value will be
        grounded. This is true whether we add an edge due to a forward
        operation, inverse operation, or conditional inverse operation.

        If adding this edge creates a grounded/ungrounded node which already
        exists, then raises a SynthError.
        """
        p = ProgramNode(fn,
                        in_values=in_nodes,
                        out_value=out_node,
                        action_num=action_num)

        # If out_node is already grounded and we would re-ground it,
        # then op is redundant
        if self.is_grounded(out_node) and self.inputs_grounded(p):
            raise SynthError('forward out already grounded')

        # If any of the ungrounded in-nodes already exist
        # then op is redundant
        # ForwardOp makes sure its input nodes are grounded, so this only
        # happens if its from an inverse/cond-inverse op.
        if any(n in self.graph.nodes and not self.is_grounded(n)
               for n in in_nodes):
            # if there is an existing program node whose inputs are these
            # nodes, then it's redundant.
            for p in self.get_program_nodes():
                if set(in_nodes) == set(p.in_values):
                    raise SynthError('existing inverse op')

            # otherwise, it's only redundant if all of the input nodes exist
            # or are already grounded.
            if all(n in self.graph.nodes for n in in_nodes):
                raise SynthError('existing inverse op 2')


        # Otherwise add edges between p and its inputs and outputs
        # ValueNodes are automatically added if they do not exist.
        # Otherwise existing value is used (based on __eq__ check).
        self.graph.add_edge(p, out_node)
        for in_node in in_nodes:
            self.graph.add_edge(in_node, p)

        # Ground output if inputs are grounded
        if self.inputs_grounded(p):
            self.ground_value_node(out_node)

        # self.check_invariants()

    def solved(self):
        """
        Returns true if we've found a program that successfully solves the
        training examples for the task embedded in this graph.end = 
        """
        return self.is_grounded(self.end)

    def get_program(self) -> Program:
        """
        If there is a program that solves the task, returns it.
        If there are multiple, just returns one of them.
        Otherwise, raises a ValueError
        """
        def find_subprogram(v: ValueNode) -> Program:
            if v in self.inputs:
                return ProgInput(self.inputs.index(v))
            if self.is_constant(v):
                return ProgConstant(v.value[0])

            valid_prog_nodes = [
                p for p in self.graph.predecessors(v)
                if self.inputs_grounded(p)
            ]
            if len(valid_prog_nodes) == 0:
                raise ValueError

            prog_node = valid_prog_nodes[0]
            subprograms = [
                find_subprogram(in_value) for in_value in prog_node.in_values
            ]
            return ProgFunction(prog_node.fn, subprograms)

        try:
            return find_subprogram(self.end)
        except ValueError:
            assert not self.solved()
            raise ValueError

    def actions_in_program(self) -> Optional[Set[int]]:
        """
        If the task was solved, returns the action steps that were used in the
        final program.
        """
        if not self.solved():
            return None

        looked_at: Set[ValueNode] = set()
        frontier = {self.end}
        actions: Set[int] = set()
        while frontier:
            assert not looked_at.intersection(frontier)
            looked_at.update(frontier)
            new_frontier = set()
            for v in frontier:
                if v in self.inputs:
                    continue
                if self.is_constant(v):
                    actions.add(self.graph.nodes[v].get("action_num"))
                    continue

                valid_prog_nodes = [
                    p for p in self.graph.predecessors(v)
                    if self.inputs_grounded(p)
                ]
                assert len(valid_prog_nodes) > 0, 'bug!'

                for p in valid_prog_nodes:
                    actions.add(p.action_num)
                    for in_value in p.in_values:
                        if in_value not in looked_at:
                            new_frontier.add(in_value)

            frontier = new_frontier

        return actions

    def draw(self):
        pos = nx.planar_layout(self.graph)
        nx.draw(
            self.graph,
            pos,
            labels={
                n: f"{n}\n g={self.is_grounded(n)}"
                for n in self.graph.nodes
            },
        )
        plt.show()
