from typing import Any, Callable, Tuple
from rl.state import State, ValueNode
from bidir.primitives.types import Grid
from bidir.primitives.functions import Function, make_function


class Op:
    def __init__(self, tp: str, arity: int, name: str = None):
        self.tp = tp
        self.arity = arity
        self.name = name

    def apply_op(self, state: State, arg_nodes: Tuple[ValueNode, ...]) -> int:
        """
        Applies the op to the graph. Returns the reward from taking this
        action.
        """
        pass


class ConstantOp(Op):
    def __init__(self, cons: Any, name: str = None):
        if name is None:
            name = str(cons)

        super().__init__(tp='constant', arity=1, name=name)
        self.cons = cons
        self.fn = Function(
            name=name,
            fn=lambda x: cons,
            # for now, take starting grid as input
            arg_types=[Grid],
            return_type=type(cons),
        )

    def apply_op(self, state: State, arg_nodes: Tuple[ValueNode, ...]) -> int:
        # for now, a constant op is a function from the start node to the
        # constant value.

        # make a value for each example
        ex_values = tuple(self.cons for _ in range(state.num_examples))
        node = ValueNode(value=ex_values)
        state.add_hyperedge(in_nodes=(state.start,), out_node=node, fn=self.fn)
        # TODO: implement rewards
        return 0


class ForwardOp(Op):
    def __init__(self, fn: Callable):
        self.fn = make_function(fn)
        super().__init__(tp='forward', arity=self.fn.arity, name=self.fn.name)

    def apply_op(self, state: State, arg_nodes: Tuple[ValueNode, ...]) -> int:
        """
        The output nodes of a forward operation will always be grounded.
        """
        arg_nodes = arg_nodes[:self.fn.arity]
        # TODO: if not, return a bad reward?
        assert all([state.is_grounded(node) for node in arg_nodes])
        # TODO: check types?

        # list containing output for each example
        out_values = []
        for i in range(state.num_examples):
            inputs = [arg.value[i] for arg in arg_nodes]
            out = self.fn.fn(*inputs)
            out_values.append(out)
        out_values = tuple(out_values)

        # forward outputs are always grounded
        out_node = ValueNode(value=out_values)
        state.add_hyperedge(in_nodes=arg_nodes, out_node=out_node, fn=self.fn)
        # TODO add rewards
        return 0


class InverseOp(Op):
    def __init__(self, forward_fn: Callable, inverse_fn: Callable):
        self.forward_fn = make_function(forward_fn)
        self.inverse_fn = inverse_fn
        super().__init__(tp='inverse',
                         arity=1,
                         name=self.forward_fn.name + '_inv')

    def apply_op(self, state: State, arg_nodes: Tuple[ValueNode, ...]) -> int:
        """
        The output node of an inverse op will always be ungrounded when first
        created (And will stay that way until all of its inputs are grounded)
        """
        out_node = arg_nodes[0]
        # TODO: return negative reward if not?
        assert not state.is_grounded(out_node)

        # gives nested tuple of shape (num_examples, num_inputs)
        in_values = tuple(self.inverse_fn(out_node.value[i])
                          for i in range(state.num_examples))

        # go to tuple of shape (num_inputs, num_examples)
        in_values = tuple(zip(*in_values))

        in_nodes = tuple(ValueNode(value) for value in in_values)

        # we just represent it in the graph
        # ...as if we had gone in the forward direction, and used the forward op
        state.add_hyperedge(in_nodes=in_nodes,
                            out_node=out_node,
                            fn=self.forward_fn)
        # TODO add reward
        return 0


class CondInverseOp(Op):
    def __init__(self, forward_fn: Callable, inverse_fn: Callable):
        self.forward_fn = make_function(forward_fn)
        super().__init__(tp='cond inverse',
                         arity=1 + self.forward_fn.arity,
                         name=self.forward_fn.name + '_cond_inv')
        # should take output and list of inputs, some of which are masks.
        # e.g. for addition: self.inverse_fn(7, [3, None]) = [3, 4]
        self.inverse_fn = inverse_fn

    def apply_op(self, state: State, arg_nodes: Tuple[ValueNode, ...]) -> int:
        out_node = arg_nodes[0]
        # TODO: give negative reward if so?
        assert not state.is_grounded(out_node)
        # args conditioned on don't need to be grounded.
        arg_nodes = arg_nodes[1:1 + self.forward_fn.arity]

        # TODO: check types?

        # gives nested list/tuple of shape (num_examples, num_inputs)
        all_arg_values = []
        for i in range(state.num_examples):
            inputs = [None if arg is None else arg.value[i]
                      for arg in arg_nodes]
            all_inputs = self.inverse_fn(out_node.value[i], inputs)
            all_arg_values.append(all_inputs)

        # go to tuple of shape (num_inputs, num_examples)
        all_arg_values = tuple(zip(*all_arg_values))

        nodes = []
        for (arg_node, arg_value) in zip(arg_nodes, all_arg_values):
            if arg_node is None:
                node = ValueNode(value=arg_value)
                nodes.append(node)
            else:
                assert arg_node.value == arg_value, (
                        'mistake made in computing cond inverse')
                nodes.append(arg_node)

        for node in nodes:
            print('node: {}'.format(node))

        state.add_hyperedge(in_nodes=tuple(nodes),
                            out_node=out_node,
                            fn=self.forward_fn)

        # TODO: add rewards
        return 0
