from typing import Dict, Tuple, Sequence, List, NamedTuple
from rl.ops.operations import Op, ForwardOp
from rl.program_search_graph import ProgramSearchGraph
from rl.environment import SynthEnvAction, SynthEnv
from bidir.task_utils import Task, twenty_four_task
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


class ProgramSpec(NamedTuple):
    task: Task
    actions: Sequence[SynthEnvAction]


class DepthOneSpec(NamedTuple):
    task: Task
    action: SynthEnvAction


def random_program(ops: Sequence[Op], inputs: Tuple,
                   depth: int) -> ProgramSpec:
    """
    Assumes task will consist of a single example.
    """
    for op in ops:
        # TODO: integrate inverse ops?
        assert isinstance(op, ForwardOp)

    program: List[SynthEnvAction] = []
    task = Task(tuple((i, ) for i in inputs), (None, ))
    SYNTH_ERROR_PENALTY = -100
    env = SynthEnv(task=task,
                   ops=ops,
                   max_actions=-1,
                   synth_error_penalty=SYNTH_ERROR_PENALTY)

    while len(program) < depth:
        action = random_action(ops, env.psg)
        _, reward, _, _ = env.step(action)

        if reward != SYNTH_ERROR_PENALTY:
            program.append(action)

    # rely on the fact that nodes are sorted by date added
    target = env.psg.get_value_nodes()[-1]

    task = Task(tuple((i, ) for i in inputs), target.value)

    rl.agent_program.rl_prog_solves(program, task, ops)

    return ProgramSpec(task, program)


def depth_one_random_sample(ops: Sequence[Op],
                            num_inputs: int,
                            max_input_int: int,
                            max_int: int = rl.ops.twenty_four_ops.MAX_INT,
                            enforce_unique: bool = False) -> DepthOneSpec:
    # currently only done for 24 game ops
    assert all(isinstance(op, ForwardOp) for op in ops)
    assert all(op.arity == 2 for op in ops)

    assert max_int >= max_input_int

    while True:
        inputs = random.sample(range(1, max_input_int + 1), k=num_inputs)
        if enforce_unique:
            inputs = sorted(inputs)  # helps keep unique
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
        return DepthOneSpec(task, action)


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
                continue

    return n
