from typing import List, Dict, Tuple, Optional

from rl.environment import ArcAction, ArcEnvObservation
from rl.operations import Op
from rl.program_search_graph import ValueNode

import random

class ArcAgent:
    """
    Base class for an Agent operating in the ArcEnvironment.
    Could be subclassed by a random agent or our NN policy.
    Feel free to change however convenient, this is just a sketch.
    """
    def __init__(self):
        pass

    def choose_action(
        self,
        obs: ArcEnvObservation,
    ) -> ArcAction:
        pass


ProgrammbleAgentProgram = List[Tuple[str, Tuple[Optional[int], ...]]]


class ProgrammableAgent(ArcAgent):
    """
    This lets you write tests to make sure the RL actions work by "programming"
    actions for each step.

    A program is a list of tuples, each of which is (op_name, arg_list).
    The first string is the op name, e.g. 'vstack_pair_cond_inv'.
    The second string is a list of arguments, e.g. '[1, 2, None]'.
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
        obs: ArcEnvObservation,
    ) -> ArcAction:
        values: List[ValueNode] = obs.psg.get_value_nodes()
        op_str, arg_node_idxs = self.program[obs.action_count]
        op = self.op_dict[op_str]
        arg_nodes = tuple(
            (None if i is None else values[i]) for i in arg_node_idxs)
        return op, arg_nodes



class RandomAgent(ArcAgent):
    """
    This guy chooses random actions in the action space
    """
    def __init__(self, op_dict: Dict[str, Op]):
        super().__init__()
        self.op_dict = op_dict

    def choose_arguments(
        self,
        op: Op,
        obs: ArcEnvObservation
    ) -> List[ValueNode]:
        """
        Returns the first argument it finds that matches the argtype
        Usually, this is the input of the grid
        So usually it'll just keep applying args to the first element
        """
        arg_nodes = []
        valuenodes = obs.psg.get_value_nodes()
        arg_nodes = []
        for argtype in op.fn.arg_types:
            arg_found = False
            for valnode in valuenodes:
                if argtype==type(valnode._value[0]):
                    # print("match between", argtype, type(valnode._value[0]))
                    arg_nodes.append(valnode)
                    arg_found = True
                    break
                else:
                    pass
                    # print("no match between", argtype, type(valnode._value[0]))
            if arg_found == False:
                raise Exception("There are no ValueNodes in the current state \
                                that could be provided as an argument to this operation.")

        return arg_nodes

    def choose_action(
        self,
        obs: ArcEnvObservation,
    ) -> ArcAction:

        # return a random op from dict
        name, op = random.choice(list(self.op_dict.items()))
        # print("name", name)
        # print("op",op)

        # pick ValueNodes to be the arguments of the op
        try: # if you could find arguments of a matching type for this op within the state, return the action
            arg_nodes = self.choose_arguments(op, obs)
            return (op, tuple(arg_nodes))
        except: # otherwise, you need to pick a new op
            return self.choose_action(obs)




class ManualAgent(ArcAgent):
    """
    This guy lets you solve arc tasks as if you were an RL agent, through the
    command line.
    """
    def __init__(self, op_dict: Dict[str, Op]):
        super().__init__()
        self.op_dict = op_dict

    def choose_action(
        self,
        obs: ArcEnvObservation,
    ) -> ArcAction:
        values: List[ValueNode] = obs.psg.get_value_nodes()
        for i, val in enumerate(values):
            print(f'{i}:\t({type(val.value[0])})\t{str(val)}')

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
                  'inverse, provide output then inputs, with masks for',
                  "unknown inputs e.g. '1, None, 2' for",
                  'vstack_pair_cond_inv')
            s = 'arg' if op.arity == 1 else 'args'
            print(f'Op chosen expects {op.arity} {s}')
            try:
                arg_choices = input('Choice: ').replace(' ', '').split(',')
                value_ixs = [
                    None if ix == 'None' else int(ix) for ix in arg_choices
                ]
            except ValueError:
                print('Non-integer index given.')
            else:
                break

        arg_nodes = [None if ix is None else values[ix] for ix in value_ixs]
        # print('arg_nodes: {}'.format(['None' if n is None else n.value[0]
        #                               for n in arg_nodes]))
        return (op, tuple(arg_nodes))
