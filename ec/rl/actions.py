from typing import List
from rl.state import State, ValueNode
from rl.operations import Op
import numpy as np


def take_action(state: State, op: Op, arg_nodes: List[ValueNode]) -> int:
    """
    Applies action to the state.
    Returns the reward received from taking this action.
    """
    if op.tp == 'forward':
        print('forward')
        return apply_forward_op(state, op, arg_nodes)
    elif op.tp == 'backward':
        out_node = arg_nodes[0]
        return apply_inverse_op(state, op, out_node)
    elif op.tp == 'link':
        out_node = arg_nodes[0]
        in_nodes = arg_nodes[1:]
        return apply_cond_inverse_op(state, op, out_node, in_nodes)


def apply_forward_op(state: State, op: Op, arg_nodes: List[ValueNode]) -> int:
    """
    The output nodes of a forward operation will always be grounded.
    """
    arg_nodes = arg_nodes[:op.fn.arity]
    assert np.all([node.is_grounded for node in arg_nodes])
    # TODO: check types?

    # works for one-arg functions.
    if op.fn.arity != 1:
        print('warning: does not work for multi-arg functions')
    valnode = arg_nodes[0]

    out_values = []
    if op.is_constant_op: # in the case of constant_ops like colors
        # ignore the arguments
        out_values = tuple(op.fn.fn() for training_example in valnode.value)
    else:
        out_values = tuple(op.fn.fn(training_example) for training_example in valnode.value)
    # print(out_values)

    # when we're doing a foreward operation, it's always going to be grounded
    out_node = ValueNode(value=out_values, is_grounded=True)
    print('out_node: {}'.format(out_node))

    # if this value node already exists, use the old object, and update it to grounded
    existing_node = state.value_node_exists(out_node)
    if existing_node is not None:
        out_node = existing_node
        out_node.is_grounded = True

    state.add_hyperedge(in_nodes=arg_nodes, out_node=out_node, fn=op.fn)
    print('state value nodes: {}'.format(state.get_value_nodes()))


def apply_inverse_op(state: State, op: Op, out_node: ValueNode) -> int:
    """
    The output node of an inverse op will always be ungrounded when first created
    (And will stay that way until all of its inputs are grounded)
    """
    assert not out_node.is_grounded

    # currently only works for one-arg functions.

    input_args = [op.inverse_fn.fn(training_example) for training_example in out_node.value]
    print(input_args)

    input_nodes = [ValueNode(value=input_args, is_grounded=False)]

    # if this value node already exists, use the old object
    # and update the output node to grounded
    existing_node = state.value_node_exists(input_nodes[0])
    if existing_node is not None:
        input_nodes[0] = existing_node
        out_node.is_grounded = True

    # we just represent it in the graph
    # ...as if we had gone in the forward direction, and used the forward op
    state.add_hyperedge(in_nodes=input_nodes, out_node=out_node, fn=op.fn)


def apply_cond_inverse_op(
    state: State,
    op: Op,
    out_node: ValueNode,
    # None in places where we want to infer input value
    arg_nodes: List[ValueNode]
) -> int:
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

    state.add_hyperedge(in_nodes=[nodes], out_node=out_node, fn=op.fn)
