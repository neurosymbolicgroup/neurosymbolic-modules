from typing import Dict, Tuple, Sequence, List
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


def random_program(ops: Sequence[Op], inputs: Tuple,
                   depth: int) -> Tuple[Task, List[SynthEnvAction]]:
    """
    Assumes task will consist of a single example.
    """
    for op in ops:
        # TODO: integrate inverse ops?
        assert isinstance(op, ForwardOp)

    program: List[SynthEnvAction] = []
    task = Task(tuple((i, ) for i in inputs), (None, ))
    SYNTH_ERROR_PENALTY = -100
    env = SynthEnv(task,
                   ops,
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

    rl.agent_program.check_rl_prog_solves(program, task, ops)

    return (task, program)


def all_depth_one_programs(
        ops: Sequence[Op],
        num_inputs: int = 5,
        max_input_int: int = 10,
        max_int: int = 100) -> List[Tuple[Task, List[SynthEnvAction]]]:
    # currently only done for 24 game ops
    assert all(isinstance(op, ForwardOp) for op in ops)
    assert all(op.arity == 2 for op in ops)

    tasks_and_programs = []
    for inputs in itertools.combinations(range(1, max_input_int + 1),
                                         num_inputs):
        for a_idx, b_idx in itertools.combinations_with_replacement(
                range(num_inputs), 2):
            for op_idx, op in enumerate(ops):
                try:
                    out = op.forward_fn.fn(inputs[a_idx], inputs[b_idx])
                except SynthError:
                    continue
                if out in inputs or out > max_int:
                    continue
                task = twenty_four_task(inputs, out)
                action = SynthEnvAction(op_idx, (a_idx, b_idx))
                tasks_and_programs.append((task, [action]))

    return tasks_and_programs


if __name__ == '__main__':
    print(len(all_depth_one_programs(rl.ops.twenty_four_ops.FORWARD_OPS,
            num_inputs=5, max_input_int=11)))

