from typing import List, Sequence, Tuple

from rl.environment import SynthEnvAction, SynthEnvObservation
from rl.ops.operations import Op
from rl.program_search_graph import ValueNode
import rl.random_programs


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


class ProgrammableAgent(SynthAgent):
    """
    This lets you write tests to make sure the RL actions work by "programming"
    actions for each step.
    """
    def __init__(self, program: Sequence[SynthEnvAction]):
        super().__init__()
        self.program = program

    def choose_action(
        self,
        obs: SynthEnvObservation,
    ) -> SynthEnvAction:
        return self.program[obs.action_count_]


class ProgrammableAgent2(SynthAgent):
    """
    Switched to having SynthEnvActions take the actual ValueNodes and Ops
    instead of indices for simply stuff with generating random programs.
    Still need this for doing tests, for when it's hard to explicitly write out
    a value node (like for ARC grids)
    """
    def __init__(self, program: Sequence[Tuple[Op, Tuple[int, ...]]]):
        super().__init__()
        self.program = program

    def choose_action(
        self,
        obs: SynthEnvObservation,
    ) -> SynthEnvAction:
        op, arg_idxs = self.program[obs.action_count_]
        nodes = obs.psg.get_value_nodes()
        args = [nodes[idx] for idx in arg_idxs]
        return SynthEnvAction(op, args)


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
        return rl.random_programs.random_action(self.ops, obs.psg)


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
                args = [values[idx] for idx in arg_idxs]
            except ValueError:
                print('Non-integer index given.')
            else:
                break

        return SynthEnvAction(op, tuple(args))
