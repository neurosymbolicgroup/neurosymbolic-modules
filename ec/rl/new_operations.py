from typing import Tuple, List, Any, Callable, Optional
from rl.new_state import State, ForwardNode, InverseNode, ValueNode
from rl.functions import Function, make_function, InverseFn, CondInverseFn
from bidir.primitives.inverse_functions import SoftTypeError
from bidir.primitives.types import Grid


class Op:
    def __init__(self, arity: int, name: str):
        self.arity = arity
        # needed for choosing from OP_DICT
        self.name = name

    def apply_op(self, state: State, arg_nodes: Tuple[Optional[ValueNode], ...]) -> int:
        pass


class ConstantOp(Op):
    def __init__(self, cons: Any, name: str = None):
        if name is None:
            name = str(cons)

        self.cons = cons
        self.fn = Function(
            name=name,
            fn=lambda x: cons,
            # for now, take starting grid as input
            # TODO get rid of input arg?
            arg_types=[Grid],
            return_type=type(cons),
        )
        # TODO make arity zero?
        super().__init__(arity=1, name=name)

    def apply_op(self, state: State, arg_nodes: Tuple[Optional[ValueNode], ...]) -> int:
        train_values = tuple(self.cons for _ in range(state.num_train))
        test_values = tuple(self.cons for _ in range(state.num_test))

        node = ForwardNode(train_values, test_values, is_constant=True)
        # TODO get rid of edge here?
        state.add_hyperedge(in_nodes=(state.start, ),
                            out_node=node,
                            fn=self.fn)
        # need to do after edge is added
        state.process_forward_node(node)
        return 0


class ForwardOp(Op):
    def __init__(self, fn: Callable):
        self.fn = make_function(fn)
        super().__init__(arity=self.fn.arity, name=self.fn.name)

    def apply_op(self, state: State, arg_nodes: Tuple[Optional[ValueNode], ...]) -> int:
        arg_nodes = arg_nodes[:self.fn.arity]
        # TODO: if not, return a bad reward?
        assert all(isinstance(node, ForwardNode) for node in arg_nodes)

        # TODO: check types?
        try:
            train_arg_values = tuple(arg.train_values for arg in arg_nodes)
            test_arg_values = tuple(arg.test_values for arg in arg_nodes)
            train_outputs = self.fn.vectorized_fn(train_arg_values)
            test_outputs = self.fn.vectorized_fn(test_arg_values)
        except SoftTypeError:
            # TODO penalize?
            return 0

        out_node = ForwardNode(train_outputs, test_outputs)
        state.add_hyperedge(in_nodes=arg_nodes, out_node=out_node, fn=self.fn)
        # need to do after, so the edge is present
        state.process_forward_node(out_node)

        return 0


class InverseOp(Op):
    def __init__(self, forward_fn: Callable, inverse_fn: Callable):
        self.fn = InverseFn(forward_fn, inverse_fn)
        super().__init__(arity=1, name = self.fn.name)

    def apply_op(self, state: State, arg_nodes: Tuple[Optional[ValueNode], ...]) -> int:
        out_node = arg_nodes[0]
        # TODO: return negative reward if not?
        assert isinstance(out_node, InverseNode)

        try:
            # gives tuple of shape (num_inputs, num_train)
            in_values = self.fn.vectorized_inverse(out_node.train_values)
        except SoftTypeError:
            # TODO give negative reward
            return 0

        # now compute the new negative values.

        # (num_inputs, List[will hold num valid new values])
        inputs_negative_values: Tuple[List[Tuple]] = tuple(
            [] for _ in range(self.arity))
        # apologies for ambiguous variable names
        for values in out_node.test_negative_values:
            try:
                # tuple of shape (num_inputs, num_test)
                inputs_negative_value = self.fn.vectorized_inverse(values)

                # each input gets its own negative value
                for negative_value, holding_list in zip(
                        inputs_negative_value, inputs_negative_values):
                    holding_list.append(negative_value)

            except SoftTypeError:
                # for each negative example set, if one of the test cases
                # doesn't work on it, then that means one of the test outputs
                # isn't possible, so it will never be created by this function.
                continue

        in_nodes = tuple(
            InverseNode(values, negative_values) for values, negative_values in
            zip(in_values, inputs_negative_values))

        state.add_hyperedge(in_nodes=in_nodes, out_node=out_node, fn=self.fn)
        # need to process after, so that the edge is in the graph
        for node in in_nodes:
            state.process_inverse_node(node)

        return 0


class CondInverseOp(Op):
    def __init__(self, forward_fn: Callable, inverse_fn: Callable):
        self.fn = CondInverseFn(forward_fn, inverse_fn)
        super().__init__(arity=1 + self.fn.forward_fn.arity, name=self.fn.name)

    def apply_op(self, state: State, arg_nodes: Tuple[Optional[ValueNode], ...]) -> int:
        out_node = arg_nodes[0]
        # TODO: give negative reward if so?
        # TODO: and penalize if InverseNode is grounded?
        assert isinstance(out_node, InverseNode)
        in_nodes = arg_nodes[1:1 + self.fn.forward_fn.arity]
        # args conditioned on need to be ForwardNodes
        # TODO give negative reward if so
        assert all(in_node is None or isinstance(in_node, ForwardNode)
                   for in_node in in_nodes)

        # TODO: check types?

        try:
            # tuple of shape (num_inputs, num_train)
            in_train_values = tuple(
                None if in_node is None else in_node.train_values
                for in_node in in_nodes
            )
            full_in_train_values = self.fn.vectorized_inverse(
                out_node.train_values, in_train_values)
        except SoftTypeError:
            # TODO give negative reward
            return 0

        # now compute the new negative values.
        # only have them for masked values - provided args remain ForwardNodes.
        in_neg_value_lists: Tuple[List[Tuple]] = tuple([]
                                                       for in_node in in_nodes)

        # neg_value is a Tuple[example] of negative values of length num_test
        in_test_values = tuple(
            None if in_node is None else in_node.test_values
            for in_node in in_nodes
        )
        for neg_out_value in out_node.test_negative_values:
            try:
                # tuple of shape (num_inputs, num_test)
                in_negative_values = self.fn.vectorized_inverse(
                    neg_out_value, in_test_values)

                # I apologize for the confusing variable names
                # we just want to distribute the negative values to each arg's
                # separate list of negative values
                for neg_value, holding_list in zip(in_negative_values,
                                                   in_neg_value_lists):
                    # later we'll ignore the lists for masked variables
                    holding_list.append(neg_value)

            except SoftTypeError:
                # ignore this example, since it won't ever be made, since the
                # output was an invalid input to this one!
                continue

        # create inverse nodes out of the collected train and negative test
        # inverses
        nodes = []
        for (in_node, train_values,
             test_negative_values) in zip(in_nodes, full_in_train_values,
                                          in_neg_value_lists):
            if in_node is None:  # was originally masked, so we found its value
                node = InverseNode(train_values, test_negative_values)
                nodes.append(node)
            else:
                # input was provided to cond inverse, check that the inverse fn
                # gave the same thing as output
                assert in_node.train_values == train_values, (
                    'mistake made in computing cond inverse')
                if len(test_negative_values) > 0:
                    assert in_node.test_values == test_negative_values[0]

                nodes.append(in_node)

        state.add_hyperedge(in_nodes=tuple(nodes),
                            out_node=out_node,
                            fn=self.fn)
        # need to do after adding hyperedge
        for node in nodes:
            if isinstance(node, InverseNode):
                state.process_inverse_node(node)
        return 0
