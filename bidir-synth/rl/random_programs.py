from typing import Dict, Tuple, Sequence, List, NamedTuple, Any
from rl.ops.operations import Op, ForwardOp
from rl.program_search_graph import ProgramSearchGraph, ValueNode
from rl.environment import SynthEnvAction, SynthEnv
from bidir.primitives.types import Grid
from bidir.task_utils import Task, twenty_four_task, arc_task, get_arc_grids
from bidir.utils import SynthError
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
        raise ValueError('No valid ops possible!')

    op_idx = random.choice(range(len(possible_ops)))
    op = ops[op_idx]

    def sample_arg(arg_type, grounded) -> int:
        return random.choice(node_dict[(arg_type, grounded)])

    arg_idxs = tuple(
        sample_arg(at, g) for (at, g) in zip(op.arg_types, op.args_grounded))

    return SynthEnvAction(op_idx, arg_idxs)


class ActionSpec(NamedTuple):
    task: Task
    action: SynthEnvAction


class ProgramSpec():
    def __init__(self, action_specs: Sequence[ActionSpec]):
        super().__init__()
        self.action_specs = action_specs
        self.task = action_specs[0].task
        self.actions = [spec.action for spec in action_specs]


def random_arc_grid() -> Grid:
    grids = get_arc_grids()  # cached, so ok to call repeatedly
    return random.choice(grids)


def get_action_specs(actions: Sequence[Tuple[int, Tuple[ValueNode, ...]]],
                     task: Task, ops: Sequence[Op]) -> Sequence[ActionSpec]:
    """
    Evaluate each action one by one. Along the way, makes ActionSpecs for each
    of them. This only works when we have ForwardOps, due to the supervised
    training setup representing the problem with a Task, not a PSG. But could
    easily be extended.
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
            # print(f"n: {n} grounded: {env.psg.is_grounded(n)}")
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


def random_program2(ops: Sequence[ForwardOp], inputs: Sequence[Any],
                    depth: int) -> ProgramSpec:
    assert len(set(inputs)) == len(inputs), 'need unique inputs'
    """
    This one actually just chooses random ops and args! Works for arity two
    functions too.
    """

    # change line 117 for getting output if not using ForwardOp's
    assert all(isinstance(op, ForwardOp) for op in ops)

    # have to store this way so we can still do them after we remove the
    # unused ops
    actions: List[Tuple[int, Tuple[ValueNode, ...]]] = []

    task = Task(tuple((inp, ) for inp in inputs), (None, ))
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
    task = Task(task.inputs, out.value)

    assert env.psg.solved()

    program = env.psg.get_program()
    print(f"program: {program}")
    assert program is not None

    # env returns a set, which probably won't be sorted!
    used_action_idxs = sorted(env.psg.actions_in_program())  # type: ignore

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
