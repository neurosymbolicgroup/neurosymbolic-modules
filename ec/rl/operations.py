import typing
from typing import Any, Callable, List, Tuple

import bidir.primitives.functions as F
from bidir.primitives.types import COLORS


class Function:
    def __init__(
        self,
        name: str,
        fn: Callable,
        arg_types: List[type],
        return_type: Any,
    ):
        self.name = name
        self.fn = fn
        self.arg_types = arg_types
        self.n_args: int = len(self.arg_types)
        self.return_type: Any = return_type

    @classmethod
    def from_typed_fn(cls, fn: Callable):
        """
        Creates a Function for the given function. Infers types from type hints,
        so the op needs to be implemented with type hints.
        """
        types: Dict[str, type] = typing.get_type_hints(fn)
        if len(types) == 0:
            raise ValueError(("Operation provided does not use type hints, "
                              "which we use when choosing actions."))

        return cls(
            name=fn.__name__,
            fn=fn,
            # list of classes, one for each input arg. skip last type (return)
            arg_types=list(types.values())[0:-1],
            return_type=types["return"],
        )


class Action:
    def __init__(self, fn: Function, eval_fn: Callable = None, tp: str):
        self.fn = fn
        # if forwards, not needed
        self.eval_fn = eval_fn
        # 'forward', 'inverse', or 'conditional inverse'
        self.tp = tp


def take_action(action: Action, arg_nodes: List[Node]):
    if action.tp == 'forward':
        take_forward_action(action, arg_nodes)
    elif action.tp == 'backward':
        take_inverse_action(action, arg_nodes[0])
    elif action.tp == 'link':
        take_conditional_inverse_action(action, arg_nodes[0], arg_nodes[1:])


def take_forward_action(action: Action, arg_nodes: List[Node]):
    assert np.all([node.grounded for node in arg_nodes])
    # TODO: check types?
    arg_values = [node.value for node in arg_nodes]
    out_value = action.fn.fn(arg_values)
    out_node = Node(value=out_value, grounded=True)
    add_hyperedge(in_nodes=arg_nodes, out_nodes=[out_node], label=action.fn)


def take_inverse_action(action: Action, out_node: Node):
    assert not out_node.grounded
    # TODO: check types?
    input_args = action.eval_fn(out_node.value)
    input_nodes = [Node(value=input_arg, grounded=False)
            for input_arg in input_args]

    add_hyperedge(in_nodes=[input_nodes], out_nodes=[out_node], label=action.fn)


def take_conditional_inverse_action(
    action: Action,
    out_node: Node,
    # None in places where we want to infer input value
    arg_nodes: List[Node]
):
    assert not out_node.grounded
    # args provided don't need to be grounded
    # TODO: check types?
    arg_values = [None if node is None else node.value for node in arg_nodes]
    all_arg_values = action.eval_fn(out_node.value, arg_values)
    nodes = []
    for (arg_node, arg_value) in zip(arc_nodes, all_arg_values):
        if arg_node is None:
            node = Node(value=arg_value, grounded=False)
            nodes.append(node)
        else:
            assert (arg_node.value == arg_value2, 
                    'mistake made in computing conditional inverse')
            nodes.append(arg_node)

    add_hyperedge(in_nodes=[nodes], out_nodes=[out_node], label=action.fn)


def forward_action(fn: Callable):
    fn = Function.from_typed_fn(fn)
    return Action(fn=fn, eval_fn=None, tp='forward')


def constant_action(value: Any):
    fn = Function(
        name=str(const),
        fn=lambda: const,
        arg_types=[],
        return_type=type(const),
    )
    return Action(fn=fn, eval_fn=None, tp='forward')


def inverse_action(fn: Function, inverse_fn: Callable):
    return Action(fn=fn, eval_fn=inverse_fn, tp='inverse')


def conditional_inverse_action(fn: Function, inverse_fn: Callable):
    return Action(fn=fn, eval_fn=inverse_fn, tp='conditional inverse')


# example setup

_FUNCTIONS: List[Callable] = [
    F.hstack_pair,
    F.hflip,
    F.vflip,
    F.vstack_pair
]

FORWARD_FUNCTION_ACTIONS = [forward_action(fn) for fn in FUNCTIONS]

COLOR_ACTIONS = [constant_action(c) for c in COLORS.ALL_COLORS]

BOOL_ACTIONS = [constant_action(b) for b in [True, False]]

# stick to small ints for now
INT_ACTIONS = [constant_action(i) for i in range(3)]

FORWARD_ACTIONS = FORWARD_FUNCTIONS + COLOR_ACTIONS + BOOL_ACTIONS + INT_ACTIONS

# sticking to one-to-one functions for now.
# TODO: Should we move these defs into bidir.primitives.functions?
_FUNCTION_INVERSE_PAIRS: List[Tuple[Callable, Callable]] = [
    (F.rotate_ccw, F.rotate_cw),
    (F.rotate_cw, F.rotate_ccw),
    (F.vflip, F.vflip),
    (F.hflip, F.hflip),
    (F.rows, F.vstack),
    (F.columns, F.hstack),
]


INVERSE_ACTIONS = [inverse_action(Function.from_typed_fn(fn), inverse) 
    for (fn, inverse) in _FUNCTION_INVERSE_PAIRS)]
