from typing import Any, Callable, Optional, Tuple

from bidir.primitives.types import Grid
from bidir.primitives.functions import Function, make_function
from rl.program_search_graph import ProgramSearchGraph, ValueNode


class Op:
    def __init__(self, arity: int, name: str):
        self.arity: int = arity
        self.name: str = name

    def apply_op(
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[Optional[ValueNode], ...],
    ) -> int:
        """
        Applies the op to the graph. Returns the reward from taking this
        action.
        """
        pass


class ConstantOp(Op):
    def __init__(self, cons: Any, name: str = None):
        if name is None:
            name = str(cons)

        super().__init__(arity=1, name=name)
        self.cons = cons
        self.fn = Function(
            name=name,
            fn=lambda x: cons,
            # for now, take starting grid as input
            arg_types=[Grid],
            return_type=type(cons),
        )

    def apply_op(  # type: ignore[override]
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[()],
    ) -> int:
        # for now, a constant op is a function from the start node to the
        # constant value.

        # make a value for each example
        ex_values = tuple(self.cons for _ in range(psg.num_examples))
        node = ValueNode(value=ex_values)
        # TODO: this edge is needed for drawing, otherwise get rid of it.
        psg.add_hyperedge(
            fn=self.fn,
            in_nodes=(psg.starts[0], ),
            out_node=node,
        )
        psg.add_constant(node)
        # TODO: implement rewards
        return 0


class ForwardOp(Op):
    def __init__(self, fn: Callable):
        self.fn = make_function(fn)
        super().__init__(arity=self.fn.arity, name=self.fn.name)

    def apply_op(  # type: ignore[override]
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[ValueNode, ...],
    ) -> int:
        """
        The output nodes of a forward operation will always be grounded.
        """
        arg_nodes = arg_nodes[:self.fn.arity]
        # TODO: if not, return a bad reward?
        assert all(psg.is_grounded(node) for node in arg_nodes)
        # TODO: check types?

        # list containing output for each example
        out_values = []
        for i in range(psg.num_examples):
            inputs = [arg.value[i] for arg in arg_nodes]
            out = self.fn.fn(*inputs)
            out_values.append(out)

        # forward outputs are always grounded
        out_node = ValueNode(value=tuple(out_values))
        psg.add_hyperedge(in_nodes=arg_nodes, out_node=out_node, fn=self.fn)
        # TODO add rewards
        return 0


class InverseOp(Op):
    """
    Apply the inverse of the invertible function
    We just store the forward_fn for reference, but don't use it
    """
    def __init__(self, forward_fn: Callable, inverse_fn: Callable):
        self.forward_fn = make_function(forward_fn) # this is just for reference, but we won't use it
        self.fn = inverse_fn

        super().__init__(arity=1, name=self.forward_fn.name + '_inv')


    def apply_op(  # type: ignore[override]
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[ValueNode],
    ) -> int:
        """
        The output node of an inverse op will always be ungrounded when first
        created (And will stay that way until all of its inputs are grounded)
        """
        out_node, = arg_nodes
        # TODO: return negative reward if not?
        assert not psg.is_grounded(out_node)

        # gives nested tuple of shape (num_examples, num_inputs)
        in_values = tuple(
            self.fn(out_node.value[i])
            for i in range(psg.num_examples))

        # go to tuple of shape (num_inputs, num_examples)
        in_values = tuple(zip(*in_values))

        in_nodes = tuple(ValueNode(value) for value in in_values)

        # we just represent it in the graph
        # ...as if we had gone in the forward direction, and used the forward op
        psg.add_hyperedge(
            fn=self.forward_fn,
            in_nodes=in_nodes,
            out_node=out_node,
        )
        # TODO add reward
        return 0


class CondInverseOp(Op):
    def __init__(self, forward_fn: Callable, inverse_fn: Callable):
        self.forward_fn = make_function(forward_fn)
        super().__init__(arity=1 + self.forward_fn.arity,
                         name=self.forward_fn.name + '_cond_inv')
        # should take output and list of inputs, some of which are masks.
        # e.g. for addition: self.inverse_fn(7, [3, None]) = [3, 4]
        self.fn = inverse_fn

    def apply_op(  # type: ignore[override]
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[Optional[ValueNode], ...],
    ) -> int:
        out_node = arg_nodes[0]
        # TODO: give negative reward if so?
        assert out_node is not None
        assert not psg.is_grounded(out_node)
        # args conditioned on don't need to be grounded.
        arg_nodes = arg_nodes[1:1 + self.forward_fn.arity]

        # TODO: check types?

        # gives nested list/tuple of shape (num_examples, num_inputs)
        all_arg_values = []
        for i in range(psg.num_examples):
            inputs = [
                None if arg is None else arg.value[i] for arg in arg_nodes
            ]
            all_inputs = self.fn(out_node.value[i], inputs)
            all_arg_values.append(all_inputs)

        # go to tuple of shape (num_inputs, num_examples)
        flipped_arg_values = tuple(zip(*all_arg_values))

        nodes = []
        for (arg_node, arg_value) in zip(arg_nodes, flipped_arg_values):
            if arg_node is None:
                node = ValueNode(value=arg_value)
                nodes.append(node)
            else:
                assert arg_node.value == arg_value, (
                    'mistake made in computing cond inverse')
                nodes.append(arg_node)

        psg.add_hyperedge(
            fn=self.forward_fn,
            in_nodes=tuple(nodes),
            out_node=out_node,
        )

        # TODO: add rewards
        return 0
