from typing import Dict, Tuple, Sequence, List, Any
from rl.ops.operations import Op, ForwardOp
from rl.program import Program
from rl.program_search_graph import ProgramSearchGraph, ValueNode
from rl.environment import SynthEnvAction, SynthEnv
from bidir.primitives.types import Grid
from bidir.primitives.functions import Function
from bidir.task_utils import Task, twenty_four_task, get_arc_grids
from bidir.utils import SynthError, assertEqual
import rl.agent_program
import rl.ops.utils
import itertools
import rl.agent_program
import rl.ops.twenty_four_ops
import random


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


class ActionSpec():
    def __init__(self,
                 task: Task,
                 action: SynthEnvAction,
                 additional_nodes: Sequence[Tuple[ValueNode, bool]] = None):
        self.task = task
        self.action = action
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


def random_arc_small_grid_inputs_sampler() -> Tuple[Tuple[Any, ...], ...]:
    max_dim=5
    grids = get_arc_grids()  # cached, so ok to call repeatedly
    small_grids = [g for g in grids if max(g.arr.shape) < max_dim]
    return ((random.choice(small_grids), ), )


def get_action_specs(actions: Sequence[Tuple[int, Tuple[ValueNode, ...]]],
                     task: Task, ops: Sequence[Op]) -> Sequence[ActionSpec]:
    """
    Evaluate each action one by one. Along the way, makes ActionSpecs for each
    of them.
    """
    SYNTH_ERROR_PENALTY = -100
    env = SynthEnv(task=task,
                   ops=ops,
                   max_actions=-1,
                   synth_error_penalty=SYNTH_ERROR_PENALTY)

    target = task.target

    action_specs = []

    done = False
    for action in actions:
        # for n in env.psg.get_value_nodes():
        #     print(f"n: {n} grounded: {env.psg.is_grounded(n)}")
        assert not done
        op_idx, args = action
        # print(f"op: {ops[op_idx]}")

        nodes = env.psg.get_value_nodes()
        input_nodes = [n for n in nodes if env.psg.is_grounded(n)]
        intermed_task = Task(tuple(n.value for n in input_nodes), target)
        # update which valuenodes we're actually looking for here
        if any(arg not in input_nodes for arg in args):
            print('hmmm')
            print(f"input_nodes: {input_nodes}")
            print(f"args: {args}")
            for action in actions:
                print(ops[action[0]])
                print(action[1])
                print()
            print(f"task: {task}")
        arg_idxs = tuple(input_nodes.index(arg) for arg in args)
        apply_arg_idxs = tuple(nodes.index(arg) for arg in args)
        intermed_action = SynthEnvAction(op_idx, arg_idxs)
        # print(f"intermed_action: {intermed_action}")

        action_specs.append(ActionSpec(intermed_task, intermed_action))

        action_to_apply = SynthEnvAction(op_idx, apply_arg_idxs)
        obs, rew, done, _ = env.step(action_to_apply)

    return action_specs


def get_action_specs2(actions: Sequence[Tuple[int, Tuple[ValueNode, ...]]],
                      task: Task, ops: Sequence[Op]) -> Sequence[ActionSpec]:
    """
    Evaluate each action one by one. Along the way, makes ActionSpecs for each
    of them.
    """
    SYNTH_ERROR_PENALTY = -100
    env = SynthEnv(task=task,
                   ops=ops,
                   max_actions=-1,
                   synth_error_penalty=SYNTH_ERROR_PENALTY)

    target = task.target

    action_specs = []

    done = False
    for action in actions:
        # for n in env.psg.get_value_nodes():
        #     print(f"n: {n} grounded: {env.psg.is_grounded(n)}")
        assert not done
        op_idx, args = action
        # print(f"op: {ops[op_idx]}")

        nodes = env.psg.get_value_nodes()
        input_nodes = [n for n in nodes if env.psg.is_grounded(n)]
        intermed_task = Task(tuple(n.value for n in input_nodes), target)
        # update which valuenodes we're actually looking for here
        if any(arg not in input_nodes for arg in args):
            print('hmmm')
            print(f"input_nodes: {input_nodes}")
            print(f"args: {args}")
            for action in actions:
                print(ops[action[0]])
                print(action[1])
                print()
            print(f"task: {task}")
        arg_idxs = tuple(input_nodes.index(arg) for arg in args)
        apply_arg_idxs = tuple(nodes.index(arg) for arg in args)
        intermed_action = SynthEnvAction(op_idx, arg_idxs)
        # print(f"intermed_action: {intermed_action}")

        action_specs.append(ActionSpec(intermed_task, intermed_action))

        action_to_apply = SynthEnvAction(op_idx, apply_arg_idxs)
        obs, rew, done, _ = env.step(action_to_apply)

    return action_specs


def bidirize_program(task: Task,
                     psg: ProgramSearchGraph,
                     ops: Sequence[Op],
                     inv_prob: float = 0.75,
                     cond_inv_prob: float = 0.75) -> ProgramSpec:

    fw_dict = rl.ops.utils.fw_dict(ops)
    inv_dict = rl.ops.utils.inv_dict(ops)
    cond_inv_dict = rl.ops.utils.cond_inv_dict(ops)

    op_to_op_idx = dict(zip(ops, range(len(ops))))
    nodes = psg.get_value_nodes()
    node_to_old_idx = dict(zip(nodes, range(len(nodes))))
    node_to_new_idx = dict(
        zip([ValueNode(i) for i in task.inputs + (task.target, )],
            range(len(task.inputs) + 1)))

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
    # print(f"inputs: {inputs}")
    task_attempts = 0
    while task_attempts < 10:
        if task_attempts > 0:
            # if this only happens once every 100 calls or so, it's fine
            print('warning: taking more tries than expected')
        task_attempts += 1
        task, psg, _, program = random_task(fw_ops,
                                            inputs,
                                            depth=depth)
        # print(f"program: {program}")
        # print(f"program: {program}")
        # print(f"task: {task}")
        if forward_only:
            inv_prob = 0.0
            cond_inv_prob = 0.0
        else:
            inv_prob = 0.8
            cond_inv_prob = 0.8

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
            except IndexError:
                # doesn't solve. this is because cond-inv-ops sometimes don't
                # work
                pass
            except AssertionError:
                # just going to assume this is ok to -- pretty sure it's also
                # from cond-inv-ops.
                pass
                # print(f"task: {task}")
                # for action_spec in bidir_prog.action_specs:
                #     action = action_spec.action
                #     print(f"action: {ops[action.op_idx], action.arg_idxs}")
                #     print(f"{action_spec.additional_nodes}")
                # raise AssertionError
            else:
                return bidir_prog


def random_program(ops: Sequence[ForwardOp], inputs: Sequence[Any],
                   depth: int) -> ProgramSpec:
    """
    This one actually just chooses random ops and args! Works for arity two
    functions too.
    Only works for ForwardOps.

    This should be deprecated, and switch to using random_bidir_program() with
    forward_only = True.
    """
    print('warning: deprecated')
    tuple_inputs = tuple((i,) for i in inputs)
    task, psg, actions, program = random_task(ops, tuple_inputs, depth)
    # print(f"program: {program}")

    assert psg.solved()

    # env returns a set, which probably won't be sorted!
    used_action_idxs = sorted(psg.actions_in_program())  # type: ignore

    used_actions = [actions[idx] for idx in used_action_idxs]
    spec = ProgramSpec(get_action_specs(used_actions, task, ops))
    assert spec.task == task

    return spec


def random_24_program(ops: Sequence[Op], inputs: Sequence[int],
                      depth: int) -> ProgramSpec:
    assert len(set(inputs)) == len(inputs), 'need unique inputs'
    """
    Instead of choosing random args for each action, make sure that the depth
    truly increases.
    """
    # print(f"inputs: {inputs}")
    assert all(isinstance(op, ForwardOp) for op in ops)
    assert all(op.arity == 2 for op in ops)

    action_specs: List[ActionSpec] = []

    task = Task(tuple((i, ) for i in inputs), (None, ))
    SYNTH_ERROR_PENALTY = -100
    env = SynthEnv(task=task,
                   ops=ops,
                   max_actions=-1,
                   synth_error_penalty=SYNTH_ERROR_PENALTY)

    # first action is truly random
    def random_first_action() -> SynthEnvAction:
        return random_action(ops, env.psg)

    # remaining actions chain on the first action
    def random_later_action() -> SynthEnvAction:
        grounded_nodes = [
            n for n in env.psg.get_value_nodes() if env.psg.is_grounded(n)
        ]

        op_idx = random.choice(range(len(ops)))
        # take the most recent output for the first arg
        # most recently added node is one of the args
        # but we add one, because grounded nodes doesn't include the target
        # node!
        arg1_idx = len(grounded_nodes) - 1 + 1
        # other arg is a random choice of the other forward nodes
        arg2_idx = random.choice(range(len(grounded_nodes)))

        arg_idxs = (arg1_idx, arg2_idx)
        if random.random() > 0.5:
            arg_idxs = (arg2_idx, arg1_idx)

        return SynthEnvAction(op_idx, arg_idxs)

    while len(action_specs) < depth:
        if len(action_specs) == 0:
            action = random_first_action()
        else:
            action = random_later_action()

        _, reward, _, _ = env.step(action)

        if reward != SYNTH_ERROR_PENALTY:
            # since we only have forward ops, task will always be a set of
            # input nodes, and the target node.
            grounded_nodes = [
                n for n in env.psg.get_value_nodes() if env.psg.is_grounded(n)
            ]

            # we already evaluated the action, so last one is the out from it
            current_inputs = grounded_nodes[:-1]
            output = grounded_nodes[-1]
            if output.value[0] == 0:
                # zeros mess stuff up, since we require the next op to use it,
                # and zero done with anything either gives an error or an
                # already existing node, so we can't come up with anything to
                # do.
                # so start over the search.
                return random_24_program(ops, inputs, depth)

            target = grounded_nodes[-1]  # the intermediate target.

            current_task = Task(tuple(i.value for i in current_inputs),
                                target.value)
            action_specs.append(ActionSpec(current_task, action))

    # now change all of the targets to be the "final target"
    target = env.psg.get_value_nodes()[-1]
    assert env.psg.is_grounded(target)

    # revise so that each step's target is the final output
    action_specs = [
        ActionSpec(Task(spec.task.inputs, target.value), spec.action)
        for spec in action_specs
    ]
    # revise original task too
    task = Task(task.inputs, target.value)

    program_spec = ProgramSpec(action_specs)
    assert rl.agent_program.rl_prog_solves(program_spec.actions, task, ops)
    return program_spec


def depth_one_random_arc_sample(ops: Sequence[ForwardOp]) -> ActionSpec:
    # if depth one, has to only take one input to start, for now
    assert all(op.arity == 1 for op in ops)
    # nothing wrong with the alternatives in principle, just sticking to this
    # for now
    assert all(op.forward_fn.arg_types == [Grid] for op in ops)
    assert all(op.forward_fn.return_type == Grid for op in ops)
    # TODO: use multiple examples!

    while True:
        input_grid = random_arc_grid()
        op_idx = random.choice(range(len(ops)))
        op = ops[op_idx]
        try:
            out = op.forward_fn.fn(input_grid)
        except SynthError:
            continue

        assert isinstance(out, Grid)
        task = Task(((input_grid, ), ), (out, ))

        action = SynthEnvAction(op_idx, (0, ))
        return ActionSpec(task, action)


def depth_one_random_24_sample(ops: Sequence[Op],
                               num_inputs: int,
                               max_input_int: int,
                               max_int: int = rl.ops.twenty_four_ops.MAX_INT,
                               enforce_unique: bool = False) -> ActionSpec:
    """
    enforce unique checks that there's only one valid solution - in case we're
    doing supervised training.
    """
    # currently only done for 24 game ops
    assert all(isinstance(op, ForwardOp) for op in ops)
    assert all(op.arity == 2 for op in ops)

    assert max_int >= max_input_int

    while True:
        inputs = random.sample(range(1, max_input_int + 1), k=num_inputs)

        op_idx = random.choice(range(len(ops)))
        op = ops[op_idx]
        if enforce_unique:
            a_idx, b_idx = random.sample(range(num_inputs), k=2)
        else:
            a_idx, b_idx = random.choices(range(num_inputs), k=2)

        try:
            out = op.forward_fn.fn(inputs[a_idx], inputs[b_idx])
        except SynthError:
            continue

        if out in inputs or out > max_int:
            continue

        task = twenty_four_task(tuple(inputs), out)

        if enforce_unique and num_depth_one_solutions(ops, task) > 1:
            continue

        action = SynthEnvAction(op_idx, (a_idx, b_idx))
        return ActionSpec(task, action)


def num_depth_one_solutions(ops: Sequence[Op], task: Task) -> int:
    # single example
    assert len(task.target) == 1

    inputs = [i[0] for i in task.inputs]
    out = task.target[0]

    n = 0

    # only works for 24 ops at the moment
    assert all(isinstance(op, ForwardOp) for op in ops)
    assert all(op.arity == 2 for op in ops)

    for (c, d) in itertools.combinations_with_replacement(inputs, 2):
        for op in ops:
            try:
                if op.forward_fn.fn(c, d) == out:
                    n += 1
            except SynthError:
                pass

            try:
                if op.forward_fn.fn(d, c) == out:
                    n += 1
            except SynthError:
                pass

    return n
