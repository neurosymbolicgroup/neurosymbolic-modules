import typing
from typing import Any, Callable, List, Dict
import numpy as np



class ValueNode:
    """
    Value nodes are what we called "input ports" and "output ports".
    They have a value, and are either grounded or not grounded.

    Values are a list of objects that the function evaluates to at that point 
    (one object for each training example)

    All the value nodes are contained inside a program node 
    (they are the input and output values to a that program node)

    All the actual edges drawn in the graph are between ValueNodes
    """
    def __init__(self, value: Any, is_grounded=False):
        self.value = value
        self.is_grounded = is_grounded

    def __str__(self):
        """
        Make the string representation just the value of the first training example
        (This function would just be for debugging purposes)
        """
        return "Val:\n" + str(self.value[0])


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
        return "Fn:\n" + str(self.fn) #+ " " + str(hash(self))[0:4]


class Function:
    def __init__(
        self,
        name: str,
        fn: Callable,
        arg_types: List[type],
        return_type: type,
    ):
        self.name = name
        self.fn = fn
        self.arg_types = arg_types
        self.arity: int = len(self.arg_types)
        self.return_type: type = return_type

    @classmethod
    def from_typed_fn(cls, fn: Callable):
        """
        Creates a Function for the given function. Infers types from type hints,
        so the op needs to be implemented with type hints.
        """
        types: Dict[str, type] = typing.get_type_hints(fn)
        if len(types) == 0:
            raise ValueError(("Operation provided does not use type hints, "
                              "which we use when choosing ops."))

        return cls(
            name=fn.__name__,
            fn=fn,
            # list of classes, one for each input arg. skip last type (return)
            arg_types=list(types.values())[0:-1],
            return_type=types["return"],
        )

    def __str__(self):
        return self.name



class Op:
    def __init__(self, fn: Function, inverse_fn: Callable, tp: str):
        self.fn = fn
        # if forwards, not needed
        self.inverse_fn = inverse_fn
        # 'forward', 'inverse', or 'cond inverse'
        self.tp = tp


def take_op(op: Op, arg_nodes: List[ValueNode]):
    if op.tp == 'forward':
        take_forward_op(op, arg_nodes)
    elif op.tp == 'backward':
        take_inverse_op(op, arg_nodes[0])
    elif op.tp == 'link':
        take_cond_inverse_op(op, arg_nodes[0], arg_nodes[1:])


def take_forward_op(state, op: Op, arg_nodes: List[ValueNode]):
    assert np.all([node.is_grounded for node in arg_nodes])
    # TODO: check types?

    # works for one-arg functions.
    valnode = arg_nodes[0]

    # separate into args
    # arg_values = [node.value for node in arg_nodes]

    #print(len(arg_values)) # number of arguments to the function

    # separate args into training examples
    # out_value = [op.fn.fn(vals) for vals in arg_values]
    # for training_example in valnode.value:
    # print(training_example)
    out_values = [op.fn.fn(training_example) for training_example in valnode.value]
    print(out_values)

    out_node = ValueNode(value=out_values, is_grounded=True)
    state.add_hyperedge(in_nodes=arg_nodes, out_nodes=[out_node], fn=op.fn)


def take_inverse_op(op: Op, out_node: ValueNode):
    assert not out_node.is_grounded
    # TODO: check types?
    input_args = op.inverse_fn(out_node.value)
    input_nodes = [ValueNode(value=input_arg, is_grounded=False)
            for input_arg in input_args]

    add_hyperedge(in_nodes=[input_nodes], out_nodes=[out_node], fn=op.fn)


def take_cond_inverse_op(
    op: Op,
    out_node: ValueNode,
    # None in places where we want to infer input value
    arg_nodes: List[ValueNode]
):
    assert not out_node.is_grounded
    # args provided don't need to be grounded!
    # TODO: check types?
    arg_values = [None if node is None else node.value for node in arg_nodes]
    all_arg_values = op.inverse_fn(out_node.value, arg_values)
    nodes = []
    for (arg_node, arg_value) in zip(arg_nodes, all_arg_values):
        if arg_node is None:
            node = ValueNode(value=arg_value, is_grounded=False)
            nodes.append(node)
        else:
            assert arg_node.value == arg_value, (
                    'mistake made in computing cond inverse')
            nodes.append(arg_node)

    add_hyperedge(in_nodes=[nodes], out_nodes=[out_node], fn=op.fn)


def forward_op(fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Op(fn=fn, inverse_fn=None, tp='forward')


def constant_op(cons: Any):
    fn = Function(
        name=str(cons),
        fn=lambda: cons,
        arg_types=[],
        return_type=type(cons),
    )
    return Op(fn=fn, inverse_fn=None, tp='forward')


def inverse_op(fn: Callable, inverse_fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Op(fn=fn, inverse_fn=inverse_fn, tp='inverse')


def cond_inverse_op(fn: Callable, inverse_fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Op(fn=fn, inverse_fn=inverse_fn, tp='cond inverse')
