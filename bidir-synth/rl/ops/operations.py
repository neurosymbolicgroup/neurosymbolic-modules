from typing import Any, Callable, Tuple, List

from bidir.utils import assertEqual
from bidir.primitives.functions import Function, make_function
from rl.program_search_graph import ProgramSearchGraph, ValueNode


class Op:
    def __init__(self, name: str, arg_types: List[type],
                 args_grounded: List[bool], forward_fn: Function):
        """
            name: name of the op.
            arg_types: List of types of the input arguments.
            args_grounded: bools for whether argument i is grounded or not.
        """
        self.arity: int = len(arg_types)
        self.name: str = name
        self.arg_types = arg_types
        assert len(self.arg_types) == self.arity
        self.args_grounded = args_grounded
        self.forward_fn = forward_fn

    def check_groundedness(self, args: Tuple[ValueNode, ...],
                           psg: ProgramSearchGraph):
        # TODO: check types too?
        # TODO: truncate args elsewhere
        # assert len(args) == self.arity
        args = args[:self.arity]
        assert all(not expects_grounded or psg.is_grounded(arg)
                   for expects_grounded, arg in zip(self.args_grounded, args))

    def apply_op(
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[ValueNode, ...],
    ):
        """
        Applies the op to the graph.
        """
        pass


class ConstantOp(Op):
    def __init__(self, cons: Any, name: str = None):
        if name is None:
            name = str(cons)

        forward_fn = Function(
            name=name,
            fn=lambda: cons,
            arg_types=[],
            return_type=type(cons),
        )
        super().__init__(name=name,
                         arg_types=[],
                         args_grounded=[],
                         forward_fn=forward_fn)
        self.cons = cons

    def apply_op(  # type: ignore[override]
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[()],
    ):
        # make a value for each example
        ex_values = tuple(self.cons for _ in range(psg.num_examples))
        node = ValueNode(value=ex_values)
        psg.add_constant(node)


class ForwardOp(Op):
    def __init__(self, fn: Callable):
        forward_fn = make_function(fn)
        arity = forward_fn.arity
        args_grounded = [True] * arity
        super().__init__(name=forward_fn.name,
                         arg_types=forward_fn.arg_types,
                         args_grounded=args_grounded,
                         forward_fn=forward_fn)

    def apply_op(  # type: ignore[override]
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[ValueNode, ...],
    ):
        self.check_groundedness(arg_nodes, psg)

        # list containing output for each example
        out_values = []
        for i in range(psg.num_examples):
            inputs = [arg.value[i] for arg in arg_nodes]
            out = self.forward_fn.fn(*inputs)
            out_values.append(out)

        # forward outputs are always grounded
        out_node = ValueNode(value=tuple(out_values))
        psg.add_hyperedge(in_nodes=arg_nodes,
                          out_node=out_node,
                          fn=self.forward_fn)


class InverseOp(Op):
    """
    Apply the inverse of the invertible function
    We store the forward_fn for reference
    """
    def __init__(self, forward_fn: Callable, inverse_fn: Callable):
        real_forward_fn = make_function(forward_fn)
        self.inverse_fn = inverse_fn

        super().__init__(name=real_forward_fn.name + '_inv',
                         arg_types=[real_forward_fn.return_type],
                         args_grounded=[False],
                         forward_fn=real_forward_fn)

    def apply_op(  # type: ignore[override]
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[ValueNode],
    ):
        """
        The output node of an inverse op will always be ungrounded when first
        created (And will stay that way until all of its inputs are grounded)
        """
        out_node, = arg_nodes
        self.check_groundedness(arg_nodes, psg)

        # gives nested tuple of shape (num_examples, num_inputs)
        in_values = tuple(
            self.inverse_fn(out_node.value[i])
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


class CondInverseOp(Op):
    def __init__(self,
                 forward_fn: Callable,
                 inverse_fn: Callable,
                 expects_cond: List[bool],
                 name=None):
        """
        expects_cond - where the inverse_fn is taking its conditional
            arguments.
        """
        real_forward_fn = make_function(forward_fn)
        # actually need to know the arity of this
        # takes in the output grid and some subset of inputs, then returns all
        # inputs as a tuple. ex: add_cond_inv(5, 3) = (2, 3)
        self.inverse_fn = make_function(inverse_fn)
        self.expects_cond = expects_cond
        args_grounded = [False] + [True] * (self.inverse_fn.arity - 1)
        arg_types = self.inverse_fn.arg_types
        assert arg_types[0] == real_forward_fn.return_type, (
            'inverse fn doesnt take output type as first input..')

        if name is None:
            name = self.inverse_fn.name
        super().__init__(name=name,
                         arg_types=arg_types,
                         args_grounded=args_grounded,
                         forward_fn=real_forward_fn)

    def apply_op(  # type: ignore[override]
        self,
        psg: ProgramSearchGraph,
        arg_nodes: Tuple[ValueNode, ...],
    ):
        self.check_groundedness(arg_nodes, psg)
        out_node = arg_nodes[0]
        # TODO: truncate elsewhere
        arg_nodes = arg_nodes[1:1 + self.inverse_fn.arity - 1]
        assert len(arg_nodes) == self.inverse_fn.arity - 1

        # gives nested list/tuple of shape (num_examples, num_inputs)
        new_arg_values = []
        for i in range(psg.num_examples):
            inputs = tuple([out_node.value[i]] +
                           [arg.value[i] for arg in arg_nodes])
            new_inputs = self.inverse_fn.fn(*inputs)
            new_arg_values.append(new_inputs)

        # go to tuple of shape (num_new_inputs, num_examples)
        flipped_arg_values = tuple(zip(*new_arg_values))

        cond_count = 0
        new_count = 0
        nodes = []
        arg_ix = 0
        # compile input values from two locations: the new computed values, and
        # the values provided as conditional inputs
        while arg_ix < self.forward_fn.arity:
            if self.expects_cond[arg_ix]:
                # value already exists.
                nodes.append(arg_nodes[cond_count])
                cond_count += 1
            else:
                # value was computed via cond inverse
                node = ValueNode(flipped_arg_values[new_count])
                nodes.append(node)
                new_count += 1

            arg_ix += 1
            assert new_count == arg_ix - cond_count

        assert cond_count == len(arg_nodes)
        assert new_count == len(flipped_arg_values)

        # list containing output for each example
        out_values = []
        for i in range(psg.num_examples):
            new_inputs = [arg.value[i] for arg in nodes]
            out = self.forward_fn.fn(*new_inputs)
            out_values.append(out)

        # evaluate produced inputs in the forward direction to check that
        # produces output!
        assertEqual(tuple(out_values), out_node.value)

        psg.add_hyperedge(
            fn=self.forward_fn,
            in_nodes=tuple(nodes),
            out_node=out_node,
        )
