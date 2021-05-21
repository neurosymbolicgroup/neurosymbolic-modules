from typing import Dict, Tuple, Sequence, List, Any
from rl.ops.operations import Op, ForwardOp
from rl.program import Program
from rl.program_search_graph import ProgramSearchGraph, ValueNode
from rl.environment import SynthEnvAction, SynthEnv
from bidir.primitives.types import Grid
from bidir.primitives.functions import Function
from bidir.task_utils import Task, get_arc_grids
from bidir.utils import assertEqual
import rl.agent_program
import rl.ops.utils
import rl.agent_program
import rl.ops.twenty_four_ops
import random


class ActionSpec():
    def __init__(self,
                 task: Task,
                 action: SynthEnvAction,
                 additional_nodes: Sequence[Tuple[ValueNode, bool]] = None):
        self.task = task
        self.action = action
        # needed for bidir programs
        self.additional_nodes = additional_nodes


class ProgramSpec():
    def __init__(self, action_specs: Sequence[ActionSpec]):
        super().__init__()
        self.action_specs = action_specs
        self.task = action_specs[0].task
        self.actions = [spec.action for spec in action_specs]


def random_arc_grid() -> Grid:
    grids = get_arc_grids()  # cached, so ok to call repeatedly
    return random.choice(grids)


def random_arc_grid_inputs_sampler() -> Tuple[Tuple[Any, ...], ...]:
    return ((random_arc_grid(), ), )


def random_twenty_four_inputs_sampler(
    max_input_int: int = 9,
    num_inputs: int = 4,
    min_input_int: int = 1,
) -> Tuple[Tuple[Any, ...], ...]:
    inputs = random.sample(range(min_input_int, max_input_int + 1),
                           k=num_inputs)
    return tuple((i, ) for i in inputs)


def random_arc_small_grid_inputs_sampler() -> Tuple[Tuple[Any, ...], ...]:
    max_dim = 5
    grids = get_arc_grids()  # cached, so ok to call repeatedly
    small_grids = [g for g in grids if max(g.arr.shape) < max_dim]
    return ((random.choice(small_grids), ), )


def random_action(ops: Sequence[Op],
                  psg: ProgramSearchGraph) -> SynthEnvAction:
    """
    Chooses a random op from those possible given the current
    value nodes and their types. For that op, chooses arguments randomly among
    those satisfying the types.
    """
    # map (type, is_grounded) to list of value nodes with that type/grounded
    # combination---nodes stored by their index in the nodes list.
    node_dict: Dict[Tuple[type, bool], List[int]] = {}

    nodes = psg.get_value_nodes()

    for node_idx, node in enumerate(nodes):
        grounded = psg.is_grounded(node)
        tp = type(node.value[0])
        try:
            node_dict[(tp, grounded)].append(node_idx)
        except KeyError:
            node_dict[(tp, grounded)] = [node_idx]

    def all_args_possible(op):
        return all((tp, ground) in node_dict
                   for (tp, ground) in zip(op.arg_types, op.args_grounded))

    possible_ops = [op for op in ops if all_args_possible(op)]
    if len(possible_ops) == 0:
        raise ValueError(f"No valid ops possible!: {nodes}"
                         + f"node_dict:: {node_dict:}")

    op_idx = random.choice(range(len(possible_ops)))
    op = ops[op_idx]

    def sample_arg(arg_type, grounded) -> int:
        return random.choice(node_dict[(arg_type, grounded)])

    arg_idxs = tuple(
        sample_arg(at, g) for (at, g) in zip(op.arg_types, op.args_grounded))

    return SynthEnvAction(op_idx, arg_idxs)


def bidirize_program(task: Task,
                     psg: ProgramSearchGraph,
                     ops: Sequence[Op],
                     inv_prob: float = 0.75,
                     cond_inv_prob: float = 0.75) -> ProgramSpec:

    assert len(ops) == len([op.name for op in ops]), 'duplicate op name'
    # we get ops based off the fn name, see where fw_dict, inv_dict, etc. are
    # called. sketchy AF
    assert len(ops) == len([op.forward_fn.name for op in ops]), 'duplicate op fn name'

    fw_dict = rl.ops.utils.fw_dict(ops)
    inv_dict = rl.ops.utils.inv_dict(ops)
    cond_inv_dict = rl.ops.utils.cond_inv_dict(ops)

    op_to_op_idx = dict(zip(ops, range(len(ops))))
    nodes = psg.get_value_nodes()
    node_to_old_idx = dict(zip(nodes, range(len(nodes))))
    node_to_new_idx = dict(
        zip([ValueNode(i) for i in task.inputs],
            range(len(task.inputs))))
    node_to_new_idx[ValueNode(task.target)] = len(task.inputs)

    grounded_nodes = [ValueNode(i) for i in task.inputs]
    additional_nodes: List[Tuple[ValueNode, bool]] = []

    def add_node(node: ValueNode, grounded: bool):
        node_to_new_idx[node] = len(node_to_new_idx)
        additional_nodes.append((node, grounded))

    # TODO: optionally do some forward steps before doing an inv op
    # this would have to incorporate the list of actions not currently provided
    # feel like it's better just to move to some sort of experience replay

    def get_prog_node(root: ValueNode):
        valid_prog_nodes = [
            p for p in psg.graph.predecessors(root) if psg.inputs_grounded(p)
        ]
        # assert one exists
        return valid_prog_nodes[0]

    def inv_prog(root: ValueNode, fn: Function,
                 children: Tuple[ValueNode]) -> List[ActionSpec]:
        # needed if doing inv or cond-inv op
        assert root in node_to_new_idx

        # make the action
        inv_op = inv_dict[fn.name]
        op_idx = op_to_op_idx[inv_op]
        action = SynthEnvAction(op_idx, (node_to_new_idx[root], ))

        # we mutate additional_nodes after the fact - danger!
        action_spec = ActionSpec(task, action, list(additional_nodes))

        # apply the inverse op: i.e. make the new target nodes
        # order needs to be the same as if we applied the op
        # this is why we do it before sorting them
        # (another leaky abstraction, code smells)
        for child in children:
            add_node(child, grounded=False)

        # sort children by date grounded (i.e. created going forward)
        sorted_children = sorted(children, key=lambda c: node_to_old_idx[c])
        recursive_solution = []
        for child in sorted_children:
            recursive_solution += bidirize_prog(child, forward_only=False)

        return [action_spec] + recursive_solution

    def cond_inv_prog(root: ValueNode, fn: Function,
                      children: Tuple[ValueNode]) -> List[ActionSpec]:
        # needed if doing inv or cond-inv op
        assert root in node_to_new_idx

        inv_ops = cond_inv_dict[fn.name]
        assertEqual(set(tuple(op.expects_cond) for op in inv_ops),
                    {(True, False), (False, True)})
        if inv_ops[0].expects_cond == [True, False]:
            op_left_first, op_right_first = inv_ops
        else:
            op_right_first, op_left_first = inv_ops

        # whichever side was created first--condition on that.
        left_child: ValueNode = children[0]
        sorted_children = sorted(children, key=lambda c: node_to_old_idx[c])
        first_child: ValueNode = sorted_children[0]
        second_child: ValueNode = sorted_children[1]

        actions = []
        # 1. make first child FW
        actions += bidirize_prog(first_child, forward_only=True)

        if first_child == left_child:
            op = op_left_first
        else:
            op = op_right_first

        op_idx = op_to_op_idx[op]
        assert first_child in node_to_new_idx
        arg_idxs = (node_to_new_idx[root], node_to_new_idx[first_child])
        action = SynthEnvAction(op_idx, arg_idxs)
        action_spec = ActionSpec(task, action, list(additional_nodes))

        # 2. apply the cond-inv-op
        add_node(second_child, grounded=False)
        actions.append(action_spec)

        # 3. make the second child bidir
        actions += bidirize_prog(second_child, forward_only=False)

        return actions

    def forward_prog(root: ValueNode, fn: Function,
                     children: Tuple[ValueNode]) -> List[ActionSpec]:
        # solve this with forward op.
        op = fw_dict[fn.name]
        op_idx = op_to_op_idx[op]

        # sort children by date grounded (i.e. created going forward)
        sorted_children = sorted(children, key=lambda c: node_to_old_idx[c])
        recursive_solution = []
        for child in sorted_children:
            prog = bidirize_prog(child, forward_only=True)
            recursive_solution += prog

        # now children are made. apply op
        assert all([c in grounded_nodes for c in children])
        arg_idxs = tuple(node_to_new_idx[c] for c in children)
        action = SynthEnvAction(op_idx, arg_idxs)
        action_spec = ActionSpec(task, action, list(additional_nodes))

        add_node(root, grounded=True)

        return recursive_solution + [action_spec]

    def bidirize_prog(root: ValueNode, forward_only: bool) -> List[ActionSpec]:

        if root in grounded_nodes:
            return []

        elif psg.is_constant(root):
            raise NotImplementedError

        prog_node = get_prog_node(root)
        fn = prog_node.fn
        children = prog_node.in_values
        if (not forward_only and fn.name in inv_dict
                and random.random() < inv_prob):
            actions = inv_prog(root, fn, children)
        elif (not forward_only and fn.name in cond_inv_dict
              and random.random() < cond_inv_prob):
            actions = cond_inv_prog(root, fn, children)
        else:
            actions = forward_prog(root, fn, children)

        grounded_nodes.append(root)
        return actions

    target = ValueNode(task.target)
    return ProgramSpec(bidirize_prog(target, forward_only=False))


def random_task(
    ops: Sequence[ForwardOp], inputs: Tuple[Tuple[Any, ...], ...], depth: int
) -> Tuple[Task, ProgramSearchGraph, List[Tuple[int, Tuple[ValueNode, ...]]], Program]:
    """
    Applies ops until we've made at least depth new grounded nodes. Then
    makes a task out of the most recently grounded node, and returns the psg
    too.
    """
    assert all(isinstance(op, ForwardOp) for op in ops)
    assert len(set(inputs)) == len(inputs), 'need unique inputs'

    # have to store this way so we can still do them after we remove the
    # unused ops
    actions: List[Tuple[int, Tuple[ValueNode, ...]]] = []

    num_examples = len(inputs[0])
    task = Task(inputs, tuple(None for _ in range(num_examples)))
    SYNTH_ERROR_PENALTY = -100
    env = SynthEnv(task=task,
                   ops=ops,
                   max_actions=-1,
                   synth_error_penalty=SYNTH_ERROR_PENALTY)

    grounded: List[ValueNode] = []  # to make flake8 happy
    while len(actions) < depth or len(grounded) < len(inputs) + depth:
        action = random_action(ops, env.psg)
        nodes = env.psg.get_value_nodes()
        args = tuple(nodes[idx] for idx in action.arg_idxs)

        _, reward, _, _ = env.step(action)

        nodes = env.psg.get_value_nodes()
        grounded = [n for n in nodes if env.psg.is_grounded(n)]

        # even if we get a synth error, still need to record, since SynthEnv
        # still logs it as an action, and we use SynthEnv to get which ops were
        # used in the final program.
        actions.append((action.op_idx, args))

    grounded = [n for n in env.psg.get_value_nodes() if env.psg.is_grounded(n)]
    out = grounded[-1]
    # bit of a hack - change the target node in the PSG
    env.psg.end = out

    program = env.psg.get_program()

    task = Task(task.inputs, out.value)
    return task, env.psg, actions, program


def random_bidir_program(ops: Sequence[Op], inputs: Tuple[Tuple[Any, ...], ...],
                         depth: int, forward_only: bool = False) -> ProgramSpec:
    fw_ops = [op for op in ops if isinstance(op, ForwardOp)]

    if forward_only:
        inv_prob = 0.0
        cond_inv_prob = 0.0
    else:
        inv_prob = 0.8
        cond_inv_prob = 0.8

    task_attempts = 0
    while task_attempts < 10:
        if task_attempts > 0:
            # if this only happens once every 100 calls or so, it's fine
            print('warning: taking more tries than expected')
        task_attempts += 1
        task, psg, _, program = random_task(fw_ops,
                                            inputs,
                                            depth=depth)

        bidir_attempts = 0
        while bidir_attempts < 10:
            bidir_attempts += 1
            bidir_prog = bidirize_program(task,
                                          psg,
                                          ops,
                                          inv_prob=inv_prob,
                                          cond_inv_prob=cond_inv_prob)
            try:
                assert rl.agent_program.rl_prog_solves(bidir_prog.actions,
                                                       task, ops)
            # doesn't solve. this is because cond-inv-ops sometimes don't
            # work
            except IndexError:
                assert cond_inv_prob > 0
                pass
            # just going to assume this is ok too -- pretty sure it's also
            # from cond-inv-ops.
            except AssertionError:
                assert cond_inv_prob > 0
                pass
            else:
                return bidir_prog

    raise RuntimeError('timed out making random bidir program')
