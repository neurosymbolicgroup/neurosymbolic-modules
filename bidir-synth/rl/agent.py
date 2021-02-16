from typing import List, Sequence, Tuple

from rl.environment import SynthEnvAction, SynthEnvObservation
from rl.ops.operations import Op
from rl.program_search_graph import ValueNode
from rl.random_programs import random_action


class SynthAgent:
    """
    Base class for an Agent operating in the SynthEnvironment.
    Could be subclassed by a random agent or our NN policy.
    Feel free to change however convenient, this is just a sketch.
    """
    def __init__(self):
        pass

    def choose_action(
        self,
        obs: SynthEnvObservation,
    ) -> SynthEnvAction:
        pass


ProgrammbleAgentProgram = List[Tuple[str, Tuple[int, ...]]]


class ProgrammableAgent(SynthAgent):
    """
    This lets you write tests to make sure the RL actions work by "programming"
    actions for each step.

    A program is a list of tuples, each of which is (op_name, arg_list).
    The first string is the op name, e.g. 'vstack_pair_cond_inv_top'.
    The second string is a list of arguments, e.g. '[1, 2]'.
    """
    def __init__(
        self,
        ops: Sequence[Op],
        program: ProgrammbleAgentProgram,
    ):
        super().__init__()
        self.ops = ops
        self.op_names = [op.name for op in self.ops]
        self.program = program

    def choose_action(
        self,
        obs: SynthEnvObservation,
    ) -> SynthEnvAction:
        op_name, arg_idxs = self.program[obs.action_count_]
        op_idx = self.op_names.index(op_name)
        return SynthEnvAction(op_idx, arg_idxs)


class RandomAgent(SynthAgent):
    """
    At each step, chooses a random op from those possible given the current
    value nodes and their types. For that op, chooses arguments randomly among
    those satisfying the types.
    """
    def __init__(self, ops: Sequence[Op]):
        super().__init__()
        self.ops = ops
        self.op_names = [op.name for op in self.ops]

    def choose_action(
        self,
        obs: SynthEnvObservation,
    ) -> SynthEnvAction:
        return random_action(self.ops, obs.psg)


class ManualAgent(SynthAgent):
    """
    This guy lets you solve arc tasks as if you were an RL agent, through the
    command line.
    """
    def __init__(self, ops: Sequence[Op]):
        super().__init__()
        self.ops = ops
        self.op_names = [op.name for op in self.ops]

    def choose_action(
        self,
        obs: SynthEnvObservation,
    ) -> SynthEnvAction:
        values: List[ValueNode] = obs.psg.get_value_nodes()
        for i, val in enumerate(values):
            ground_string = "G" if obs.psg.is_grounded(val) else "UG"
            print(f'{i}:\t({ground_string}) {type(val.value[0])})\t{str(val)}')

        while True:
            print("Choose an op (provided as string, e.g.",
                  "'vstack_pair_cond_inv')")
            op_name = input('Choice: ')
            if op_name in self.op_names:
                op_idx = self.op_names.index(op_name)
                op = self.ops[op_idx]
                break
            else:
                print('Invalid op given. Options: ', list(self.op_names))

        while True:
            print('Args for op, as index of value list printed. If cond.',
                  'inverse, provide output then inputs')
            s = 'arg' if op.arity == 1 else 'args'
            print(f'Op chosen expects {op.arity} {s}')
            try:
                arg_choices = input('Choice: ').replace(' ', '').split(',')
                arg_idxs = [int(idx) for idx in arg_choices]
            except ValueError:
                print('Non-integer index given.')
            else:
                break

        return SynthEnvAction(op_idx, tuple(arg_idxs))
