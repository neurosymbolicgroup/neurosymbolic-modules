from typing import List, Dict, Tuple, Optional

from rl.environment import ArcEnvObservation
from rl.operations import Op
from rl.program_search_graph import ValueNode


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
    ) -> Tuple[Op, Tuple[ValueNode]]:
        pass


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
        program: List[Tuple[str, Tuple[Optional[int]]]],
    ):
        super().__init__()
        self.op_dict = op_dict
        self.program = program

    def choose_action(
        self,
        obs: ArcEnvObservation,
    ) -> Tuple[Op, Tuple[ValueNode]]:
        values: List[ValueNode] = obs.psg.get_value_nodes()
        op_str, arg_node_idxs = self.program[obs.action_count]
        op = self.op_dict[op_str]
        arg_nodes = tuple(
            (None if i is None else values[i]) for i in arg_node_idxs)
        return op, arg_nodes


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
    ) -> Tuple[Op, Tuple[ValueNode]]:
        values: List[ValueNode] = obs.psg.get_value_nodes()
        for i, val in enumerate(values):
            print(f'{i}:\t({type(val.value[0])})\t{str(val)}')

        while True:
            print("Choose an op (provided as string, e.g.",
                  "'vstack_pair_cond_inv')")
            op = input('Choice: ')
            if op in self.op_dict:
                op = self.op_dict[op]
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
            value_ixs = input('Choice: ')
            value_ixs = value_ixs.replace(' ', '')
            value_ixs = value_ixs.split(',')
            try:
                value_ixs = [
                    None if ix == 'None' else int(ix) for ix in value_ixs
                ]
            except ValueError:
                print('Non-integer index given.')
            else:
                break

        arg_nodes = [None if ix is None else values[ix] for ix in value_ixs]
        # print('arg_nodes: {}'.format(['None' if n is None else n.value[0]
        #                               for n in arg_nodes]))
        return (op, tuple(arg_nodes))
