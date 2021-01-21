import typing
from typing import Any, Callable, List, Dict
import numpy as np

# from state_interface import add_hyperedge, update_groundedness


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


# class Op:
#     def __init__(self, fn: Function, inverse_fn: Callable, tp: str):
#         self.fn = fn
#         # if forwards, not needed
#         self.inverse_fn = inverse_fn
#         # 'forward', 'inverse', or 'cond inverse'
#         self.tp = tp


# def take_op(op: Op, arg_nodes: List[ValueNode]):
#     if op.tp == 'forward':
#         take_forward_op(op, arg_nodes)
#     elif op.tp == 'backward':
#         take_inverse_op(op, arg_nodes[0])
#     elif op.tp == 'link':
#         take_cond_inverse_op(op, arg_nodes[0], arg_nodes[1:])


# def take_forward_op(op: Op, arg_nodes: List[ValueNode]):
#     assert np.all([node.is_grounded for node in arg_nodes])
#     # TODO: check types?
#     arg_values = [node.value for node in arg_nodes]
#     out_value = op.fn.fn(arg_values)
#     out_node = ValueNode(value=out_value, is_grounded=True)
#     add_hyperedge(in_nodes=arg_nodes, out_nodes=[out_node], fn=op.fn)


# def take_inverse_op(op: Op, out_node: ValueNode):
#     assert not out_node.is_grounded
#     # TODO: check types?
#     input_args = op.inverse_fn(out_node.value)
#     input_nodes = [ValueNode(value=input_arg, is_grounded=False)
#             for input_arg in input_args]

#     add_hyperedge(in_nodes=[input_nodes], out_nodes=[out_node], fn=op.fn)


# def take_cond_inverse_op(
#     op: Op,
#     out_node: ValueNode,
#     # None in places where we want to infer input value
#     arg_nodes: List[ValueNode]
# ):
#     assert not out_node.is_grounded
#     # args provided don't need to be grounded!
#     # TODO: check types?
#     arg_values = [None if node is None else node.value for node in arg_nodes]
#     all_arg_values = op.inverse_fn(out_node.value, arg_values)
#     nodes = []
#     for (arg_node, arg_value) in zip(arg_nodes, all_arg_values):
#         if arg_node is None:
#             node = ValueNode(value=arg_value, is_grounded=False)
#             nodes.append(node)
#         else:
#             assert arg_node.value == arg_value, (
#                     'mistake made in computing cond inverse')
#             nodes.append(arg_node)

#     add_hyperedge(in_nodes=[nodes], out_nodes=[out_node], fn=op.fn)


# def forward_op(fn: Callable):
#     fn = Function.from_typed_fn(fn)
#     return Op(fn=fn, inverse_fn=None, tp='forward')


# def constant_op(cons: Any):
#     fn = Function(
#         name=str(cons),
#         fn=lambda: cons,
#         arg_types=[],
#         return_type=type(cons),
#     )
#     return Op(fn=fn, inverse_fn=None, tp='forward')


# def inverse_op(fn: Callable, inverse_fn: Callable):
#     fn = Function.from_typed_fn(fn)
#     return Op(fn=fn, inverse_fn=inverse_fn, tp='inverse')


# def cond_inverse_op(fn: Callable, inverse_fn: Callable):
#     fn = Function.from_typed_fn(fn)
#     return Op(fn=fn, inverse_fn=inverse_fn, tp='cond inverse')
