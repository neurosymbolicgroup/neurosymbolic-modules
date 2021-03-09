from typing import Dict, Tuple, Sequence, List, NamedTuple
from rl.ops.operations import Op, ForwardOp
from rl.program_search_graph import ProgramSearchGraph, ValueNode
from rl.environment import SynthEnvAction, SynthEnv
from bidir.primitives.types import Grid
from bidir.task_utils import Task, twenty_four_task, arc_task
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
    node_dict: Dict[Tuple[type, bool], List[ValueNode]] = {}

    nodes = psg.get_value_nodes()

    for node in nodes:
        grounded = psg.is_grounded(node)
        tp = type(node.value[0])
        try:
            node_dict[(tp, grounded)].append(node)
        except KeyError:
            node_dict[(tp, grounded)] = [node]

    def all_args_possible(op):
        return all((tp, ground) in node_dict
                   for (tp, ground) in zip(op.arg_types, op.args_grounded))

    possible_ops = [op for op in ops if all_args_possible(op)]
    if len(possible_ops) == 0:
        raise ValueError('No valid ops possible!')

    op = random.choice(ops)

    def sample_arg(arg_type, grounded) -> ValueNode:
        return random.choice(node_dict[(arg_type, grounded)])

    args = tuple(
        sample_arg(at, g) for (at, g) in zip(op.arg_types, op.args_grounded))

    return SynthEnvAction(op, args)


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
    task_num = random.choice(range(400))
    train = random.random() > 0.5
    inputs = arc_task(task_num, train).inputs
    assert len(inputs) == 1  # each task only has one input, unlike 24
    random_example = random.choice(inputs[0])
    assert isinstance(random_example, Grid)

    return random_example


def get_action_specs(actions: Sequence[SynthEnvAction], task: Task) -> Sequence[ActionSpec]:
    """
    Evaluate each action one by one. Along the way, makes ActionSpecs for each
    of them. This only works when we have ForwardOps, due to the supervised
    training setup representing the problem with a Task, not a PSG. But could
    easily be extended.
    """
    SYNTH_ERROR_PENALTY = -100
    env = SynthEnv(task, max_actions=-1,
                   synth_error_penalty=SYNTH_ERROR_PENALTY)

    target = task.target

    action_specs = []

    for action in actions:
        values = env.psg.get_value_nodes()
        task = Task(tuple(v.value for v in values), target)
        action_specs.append(ActionSpec(task, action))

        obs, rew, done, _ = env.step(action)
        assert env.psg.get_value_nodes()[-1] == target

    return action_specs


def random_arc_program(ops: Sequence[ForwardOp], inputs: Sequence[Grid],
                       depth: int) -> ProgramSpec:
    """
    This one actually just chooses random ops and args! Works for arity two
    functions too.
    """
    # print(f"inputs: {inputs}")
    # change line 117 for getting output if not using ForwardOp's
    assert all(isinstance(op, ForwardOp) for op in ops)

    actions: List[SynthEnvAction] = []

    task = Task(tuple((grid, ) for grid in inputs), (None, ))
    SYNTH_ERROR_PENALTY = -100
    env = SynthEnv(task=task,
                   max_actions=-1,
                   synth_error_penalty=SYNTH_ERROR_PENALTY)

    while len(actions) < depth:
        action = random_action(ops, env.psg)

        _, reward, _, _ = env.step(action)

        if reward != SYNTH_ERROR_PENALTY:
            actions.append(action)
            out = env.psg.get_value_nodes()[-1]

    # bit of a hack - change the target node in the PSG
    env.psg.end = out
    task = Task(task.inputs, out.value)

    assert env.psg.solved()
    program = env.psg.get_program()
    assert program is not None
    used_action_idxs = env.psg.actions_in_program()
    assert used_action_idxs is not None
    used_actions = [actions[idx] for idx in used_action_idxs]
    assert rl.agent_program.rl_prog_solves(used_actions, task)
    spec = ProgramSpec(get_action_specs(actions, task))
    assert spec.task == task
    return spec


def random_24_program(ops: Sequence[Op], inputs: Sequence[int],
                      depth: int) -> ProgramSpec:
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

        op = random.choice(ops)
        # take the most recent output for the first arg
        # most recently added node is one of the args
        # but we add one, because grounded nodes doesn't include the target
        # node!
        arg1 = grounded_nodes[-1]
        # other arg is a random choice of the other forward nodes
        arg2 = random.choice(grounded_nodes)

        args = (arg1, arg2)
        if random.random() > 0.5:
            args = (arg2, arg1)

        return SynthEnvAction(op, args)

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
    assert rl.agent_program.rl_prog_solves(program_spec.actions, task)
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
        op = random.choice(ops)
        try:
            out = op.forward_fn.fn(input_grid)
        except SynthError:
            continue

        assert isinstance(out, Grid)
        task = Task(((input_grid, ), ), (out, ))
        action = SynthEnvAction(op, [ValueNode((out, ))])

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

        op = random.choice(ops)
        if enforce_unique:
            a, b = random.sample(inputs, k=2)
        else:
            a, b = random.choices(inputs, k=2)

        try:
            out = op.forward_fn.fn(a, b)
        except SynthError:
            continue

        if out in inputs or out > max_int:
            continue

        task = twenty_four_task(tuple(inputs), out)

        if enforce_unique and num_depth_one_solutions(ops, task) > 1:
            continue

        action = SynthEnvAction(op, (ValueNode((a, )), ValueNode((b, ))))
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
