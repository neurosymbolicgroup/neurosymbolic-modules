import typing
from typing import Any, Callable, List, Tuple

import bidir.primitives.functions as F
import bidir.primitives.inverse_functions as F2
from bidir.primitives.types import COLORS


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

class Op:
    def __init__(self, fn: Function, inverse_fn: Callable = None, tp: str):
        self.fn = fn
        # if forwards, not needed
        self.inverse_fn = inverse_fn
        # 'forward', 'inverse', or 'cond inverse'
        self.tp = tp


def take_op(op: Op, arg_nodes: List[Node]):
    if op.tp == 'forward':
        take_forward_op(op, arg_nodes)
    elif op.tp == 'backward':
        take_inverse_op(op, arg_nodes[0])
    elif op.tp == 'link':
        take_cond_inverse_op(op, arg_nodes[0], arg_nodes[1:])


def take_forward_op(op: Op, arg_nodes: List[Node]):
    assert np.all([node.grounded for node in arg_nodes])
    # TODO: check types?
    arg_values = [node.value for node in arg_nodes]
    out_value = op.fn.fn(arg_values)
    out_node = Node(value=out_value, grounded=True)
    add_hyperedge(in_nodes=arg_nodes, out_nodes=[out_node], label=op.fn)


def take_inverse_op(op: Op, out_node: Node):
    assert not out_node.grounded
    # TODO: check types?
    input_args = op.inverse_fn(out_node.value)
    input_nodes = [Node(value=input_arg, grounded=False)
            for input_arg in input_args]

    add_hyperedge(in_nodes=[input_nodes], out_nodes=[out_node], label=op.fn)


def take_cond_inverse_op(
    op: Op,
    out_node: Node,
    # None in places where we want to infer input value
    arg_nodes: List[Node]
):
    assert not out_node.grounded
    # args provided don't need to be grounded!
    # TODO: check types?
    arg_values = [None if node is None else node.value for node in arg_nodes]
    all_arg_values = op.inverse_fn(out_node.value, arg_values)
    nodes = []
    for (arg_node, arg_value) in zip(arc_nodes, all_arg_values):
        if arg_node is None:
            node = Node(value=arg_value, grounded=False)
            nodes.append(node)
        else:
            assert (arg_node.value == arg_value2,
                    'mistake made in computing cond inverse')
            nodes.append(arg_node)

    add_hyperedge(in_nodes=[nodes], out_nodes=[out_node], label=op.fn)


def forward_op(fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Op(fn=fn, inverse_fn=None, tp='forward')


def constant_op(value: Any):
    fn = Function(
        name=str(const),
        fn=lambda: const,
        arg_types=[],
        return_type=type(const),
    )
    return Op(fn=fn, inverse_fn=None, tp='forward')


def inverse_op(fn: Callable, inverse_fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Op(fn=fn, inverse_fn=inverse_fn, tp='inverse')


def cond_inverse_op(fn: Callable, inverse_fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Op(fn=fn, inverse_fn=inverse_fn, tp='cond inverse')


_FUNCTIONS: List[Callable] = [
    F.hstack_pair,
    F.hflip,
    F.vflip,
    F.vstack_pair,
]

FORWARD_FUNCTION_OPS = [forward_op(fn) for fn in FUNCTIONS]

COLOR_OPS = [constant_op(c) for c in COLORS.ALL_COLORS]

BOOL_OPS = [constant_op(b) for b in [True, False]]

# stick to small ints for now
INT_OPS = [constant_op(i) for i in range(3)]

FORWARD_OPS = FORWARD_FUNCTIONS + COLOR_OPS + BOOL_OPS + INT_OPS

# sticking to one-to-one functions for now.
# TODO: Should we move these defs into bidir.primitives.functions?
_FUNCTION_INVERSE_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.rotate_ccw, F.rotate_cw),
    (F.rotate_cw, F.rotate_ccw),
    (F.vflip, F.vflip),
    (F.hflip, F.hflip),
    (F.rows, F.vstack),
    (F.columns, F.hstack),
    (F.block, F2.block_inv),
]


INVERSE_OPS = [inverse_op(fn, inverse)
    for (fn, inverse) in _FUNCTION_INVERSE_PAIRS)]

_FUNCTION_COND_INVERSE_PAIRS: List[Tuple[Callable, Callable]] = [
        (F.vstack_pair, F2.vstack_pair_inv),
        (F.hstack_pair, F2.hstack_pair_inv),
]

COND_INVERSE_OPS = [cond_inverse_aop(fn, inverse_fn)
    for (fn, inverse) in _FUNCTION_COND_INVERSE_OPS]

ALL_OPS = FORWARD_OPS + INVERSE_OPS + COND_INVERSE_OPS

# an op will be a choice of OP along with at most MAX_ARITY arguments
MAX_ARITY = max(op.fn.arity for op in FORWARD_OPS)
N_OPS = len(ALL_OPS)
