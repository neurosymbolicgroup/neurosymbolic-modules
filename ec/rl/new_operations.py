from typing import Tuple, List, Any, Callable
from rl.new_state import State, ForwardNode, InverseNode, ValueNode
from bidir.primitives.functions import Function, make_function
from bidir.primitives.inverse_functions import SoftTypeError
from bidir.primitives.types import Grid


class Op:
    def __init__(self, tp: str, arity: int, fn: Function, name: str = None):
        self.tp = tp
        self.arity = arity
        self.fn = fn
        self.name = fn.name if name is None else name

    def apply_op(self, state: 'State', arg_nodes: Tuple[ValueNode]) -> int:
        pass


class ConstantOp(Op):
    def __init__(self, cons: Any, name: str = None):
        if name is None:
            name = str(cons)

        self.cons = cons
        fn = Function(
            name=name,
            fn=lambda x: cons,
            # for now, take starting grid as input
            # TODO get rid of input arg?
            arg_types=[Grid],
            return_type=type(cons),
        )
        super().__init__(tp='constant', arity=1, fn=fn, name=name)

    def apply_op(self, state: State, arg_nodes: Tuple[ValueNode, ...]) -> int:
        train_values = tuple(self.cons for _ in range(state.num_train))
        test_values = tuple(self.cons for _ in range(state.num_test))

        node = ForwardNode(train_values, test_values, is_constant=True)
        # TODO get rid of edge here?
        state.add_hyperedge(in_nodes=(state.start,), out_node=node, op=self)
        # need to do after edge is added
        state.process_forward_node(node)
        return 0


class ForwardOp(Op):
    def __init__(self, fn: Callable):
        fn = make_function(fn)
        super().__init__(tp='forward', arity=fn.arity, fn=fn)

    def vectorized_fn(self,
                      arg_nodes: Tuple[ForwardNode, ...],
                      use_test: bool = False) -> Tuple:
        """
        Applies the function to a set of forward nodes at once.
        Raises a SoftTypeError if the inputs are invalid types.
        """

        # gets the ith arg from each node thats an input
        def ith_args(args: Tuple[ValueNode, ...], i: int) -> List:
            if use_test:
                return [arg.test_values[i] for arg in args]
            else:
                return [arg.train_values[i] for arg in args]

        if use_test:
            num_examples = len(arg_nodes[0].test_values)
        else:
            num_examples = len(arg_nodes[0].train_values)

        outputs = tuple(
            self.fn.fn(*ith_args(arg_nodes, i)) for i in range(num_examples))
        return outputs

    def apply_op(self, state: State, arg_nodes: Tuple[ValueNode, ...]) -> int:
        """
        The output nodes of a forward operation will always be grounded.
        """
        arg_nodes = arg_nodes[:self.fn.arity]
        # TODO: if not, return a bad reward?
        assert all(isinstance(node, ForwardNode) for node in arg_nodes)

        # TODO: check types?
        try:
            train_outputs = self.vectorized_fn(arg_nodes, use_test=False)
            test_outputs = self.vectorized_fn(arg_nodes, use_test=True)
        except SoftTypeError:
            # TODO penalize?
            return 0

        out_node = ForwardNode(train_outputs, test_outputs)
        state.add_hyperedge(in_nodes=arg_nodes, out_node=out_node, op=self)
        # need to do after, so the edge is present
        state.process_forward_node(out_node)

        return 0


class InverseOp(Op):
    def __init__(self, forward_fn: Callable, inverse_fn: Callable):
        fn = make_function(forward_fn)
        self.inverse_fn = inverse_fn
        super().__init__(tp='inverse', arity=1, fn=fn, name=fn.name + '_inv')

    def vectorized_inverse(self, out_values: Tuple) -> Tuple[Tuple]:
        """
        Given the out values, produces a tuple of shape (num_inputs,
        num_examples) by calculating the inverse fn for each value.

        Raises a SoftTypeError if one of the outputs isn't valid to
        be inverted, or if the mask of None's isn't correct.
        """
        in_values = tuple(
            self.inverse_fn(out_value) for out_value in out_values)
        # go to tuple of shape (num_inputs, num_examples)
        in_values = tuple(zip(*in_values))
        return in_values

    def apply_op(self, state: State, arg_nodes: Tuple[ValueNode, ...]) -> int:
        """
        The output node of an inverse op will always be ungrounded when first
        created (And will stay that way until all of its inputs are grounded)
        """
        out_node = arg_nodes[0]
        # TODO: return negative reward if not?
        assert isinstance(out_node, InverseNode)

        try:
            # gives tuple of shape (num_inputs, num_train)
            in_values = self.vectorized_inverse(out_node.train_values)
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
                inputs_negative_value = self.vectorized_inverse(values)

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


        state.add_hyperedge(in_nodes=in_nodes, out_node=out_node, op=self)
        # need to process after, so that the edge is in the graph
        for node in in_nodes:
            state.process_inverse_node(node)

        return 0


class CondInverseOp(Op):
    def __init__(self, forward_fn: Callable, inverse_fn: Callable):
        fn = make_function(forward_fn)
        super().__init__(tp='cond inverse',
                         arity=1 + fn.arity,
                         fn=fn,
                         name=fn.name + '_cond_inv')
        # should take output and list of inputs, some of which are masks.
        # e.g. for addition: self.inverse_fn(7, [3, None]) = [3, 4]
        self.inverse_fn = inverse_fn

    def vectorized_inverse(self,
                           out_values: Tuple,
                           arg_nodes: Tuple[ForwardNode, ...],
                           use_test: bool = False) -> Tuple[Tuple]:
        """
        Given input nodes (some of which are None) and the out value (a tuple
        of length num_examples) produces a tuple of shape (num_inputs,
        num_examples).

        If use_test is true, uses the test values from the input nodes. (This
        would be done when calculating inverses of negative values.)

        Raises a SoftTypeError if one of the outputs isn't valid to
        be inverted, or if the mask of None's isn't correct.
        """
        arg_values = []

        def get_value(arg: ForwardNode, example_i: int, use_test: bool):
            if arg is None:
                return None
            if use_test:
                return arg.test_values[example_i]
            else:
                return arg.train_values[example_i]

        if use_test:
            num_examples = len(arg_nodes[0].test_values)
        else:
            num_examples = len(arg_nodes[0].train_values)
        for i in range(num_examples):
            inputs = [get_value(arg, i, use_test) for arg in arg_nodes]
            # tuple of length num_examples
            arg_example_i_values = self.inverse_fn(out_values[i], inputs)
            # TODO assert that we get same thing back for conditioned
            # inputs?
            arg_values.append(arg_example_i_values)

        # go to tuple of shape (num_inputs, num_examples)
        flipped_arg_values = tuple(zip(*arg_values))
        return flipped_arg_values

    def apply_op(self, state: State, arg_nodes: Tuple[ValueNode, ...]) -> int:
        out_node = arg_nodes[0]
        # TODO: give negative reward if so?
        # TODO: and penalize if InverseNode is grounded?
        assert isinstance(out_node, InverseNode)
        arg_nodes = arg_nodes[1:1 + self.fn.arity]
        # args conditioned on need to be ForwardNodes
        # TODO give negative reward if so
        assert all(arg_node is None or isinstance(arg_node, ForwardNode)
                   for arg_node in arg_nodes)

        # TODO: check types?

        try:
            # tuple of shape (num_inputs, num_train)
            arg_train_values = self.vectorized_inverse(out_node.train_values,
                                                        arg_nodes,
                                                        use_test=False)
        except SoftTypeError:
            # TODO give negative reward
            return 0

        # now compute the new negative values.
        # only have them for masked values - provided args remain ForwardNodes.
        arg_negative_values: Tuple[List[Tuple]] = tuple(
            [] for arg in arg_nodes)

        # value is a Tuple[example] of negative values of length num_test
        for value in out_node.test_negative_values:
            try:
                # tuple of shape (num_inputs, num_test)
                arg_negative_value = self.vectorized_inverse(value,
                                                              arg_nodes,
                                                              use_test=True)

                # I apologize for the confusing variable names
                # we just want to distribute the negative values to each arg's
                # separate list of negative values
                for negative_value, holding_list in zip(
                        arg_negative_value, arg_negative_values):
                    holding_list.append(negative_value)

            except SoftTypeError:
                # ignore this example, since it won't ever be made, since the
                # output was an invalid input to this one!
                continue

        nodes = []
        for (arg_node, train_values,
             test_negative_values) in zip(arg_nodes, arg_train_values,
                                           arg_negative_values):
            if arg_node is None:
                node = InverseNode(train_values, test_negative_values)
                nodes.append(node)
            else:
                assert arg_node.train_values == train_values, (
                    'mistake made in computing cond inverse')
                nodes.append(arg_node)

        state.add_hyperedge(in_nodes=tuple(nodes), out_node=out_node, op=self)
        # need to do after adding hyperedge
        for node in nodes:
            if isinstance(node, InverseNode):
                state.process_inverse_node(node)
        return 0
