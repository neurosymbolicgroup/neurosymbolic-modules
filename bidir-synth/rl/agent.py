from typing import List, Dict, Tuple

from rl.environment import SynthAction, SynthEnvObservation
from rl.operations import Op
from rl.program_search_graph import ValueNode

import random


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
    ) -> SynthAction:
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
        op_dict: Dict[str, Op],
        program: ProgrammbleAgentProgram,
    ):
        super().__init__()
        self.op_dict = op_dict
        self.program = program

    def choose_action(
        self,
        obs: SynthEnvObservation,
    ) -> SynthAction:
        values: List[ValueNode] = obs.psg.get_value_nodes()
        op_str, arg_node_idxs = self.program[obs.action_count]
        op = self.op_dict[op_str]
        arg_nodes = tuple(values[i] for i in arg_node_idxs)
        return op, arg_nodes


class RandomAgent(SynthAgent):
    """
    This guy chooses random actions in the action space
    """
    def __init__(self, ops: List[Op]):
        super().__init__()
        self.ops = ops

    def choose_action(
        self,
        obs: SynthEnvObservation,
    ) -> SynthAction:

        node_dict: Dict[Tuple[type, bool], List[ValueNode]] = {}

        nodes = obs.psg.get_value_nodes()
        random.shuffle(nodes)

        for node in nodes:
            grounded = obs.psg.is_grounded(node)
            tp = type(node.value[0])
            try:
                node_dict[(tp, grounded)].append(node)
            except KeyError:
                node_dict[(tp, grounded)] = [node]

        def all_args_possible(op):
            return all((tp, ground) in node_dict
                       for (tp, ground) in zip(op.arg_types, op.args_grounded))

        possible_ops = [op for op in self.ops if all_args_possible(op)]
        # print(f"possible_ops: {[o.name for o in possible_ops]}")
        assert len(possible_ops) > 0, 'no valid ops possible!!'

        op = random.choice(possible_ops)

        def sample_arg(arg_type, grounded):
            return random.choice(node_dict[(arg_type, grounded)])

        args = tuple(
            sample_arg(at, g)
            for (at, g) in zip(op.arg_types, op.args_grounded))

        return (op, args)


class ManualAgent(SynthAgent):
    """
    This guy lets you solve arc tasks as if you were an RL agent, through the
    command line.
    """
    def __init__(self, op_dict: Dict[str, Op]):
        super().__init__()
        self.op_dict = op_dict

    def choose_action(
        self,
        obs: SynthEnvObservation,
    ) -> SynthAction:
        values: List[ValueNode] = obs.psg.get_value_nodes()
        for i, val in enumerate(values):
            ground_string = "G" if obs.psg.is_grounded(val) else "UG"
            print(f'{i}:\t({ground_string}) {type(val.value[0])})\t{str(val)}')

        while True:
            print("Choose an op (provided as string, e.g.",
                  "'vstack_pair_cond_inv')")
            op_name = input('Choice: ')
            if op_name in self.op_dict:
                op = self.op_dict[op_name]
                break
            else:
                print('Invalid op given. Options: ', list(self.op_dict.keys()))

        while True:
            print('Args for op, as index of value list printed. If cond.',
                  'inverse, provide output then inputs')
            s = 'arg' if op.arity == 1 else 'args'
            print(f'Op chosen expects {op.arity} {s}')
            try:
                arg_choices = input('Choice: ').replace(' ', '').split(',')
                value_ixs = [int(ix) for ix in arg_choices]
            except ValueError:
                print('Non-integer index given.')
            else:
                break

        arg_nodes = [values[ix] for ix in value_ixs]
        return (op, tuple(arg_nodes))
