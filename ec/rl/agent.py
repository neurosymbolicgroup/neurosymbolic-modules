from rl.operations import Op
from rl.state import State, ValueNode
from typing import List, Tuple, Dict


class ArcAgent:
    """
    Base class for an Agent operating in the ArcEnvironment.
    Could be subclassed by a random agent or our NN policy.
    Feel free to change however convenient, this is just a sketch.
    """
    def __init__(self, ops: List[Op], arity: int):
        self.ops = ops
        self.arity = arity

    def choose_action(self, state: State) -> Tuple[Op, List[ValueNode]]:
        pass


class ManualAgent(ArcAgent):
    def __init__(self, ops: List[Op], arity: int, op_dict: Dict[str, Op]):
        super().__init__(ops, arity)
        self.op_dict = op_dict

    def choose_action(self, state: State) -> Tuple[Op, List[ValueNode]]:
        values: List[ValueNode] = state.get_value_nodes()
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
                value_ixs = [None if ix == 'None' else int(ix)
                             for ix in value_ixs]
            except ValueError:
                print('Non-integer index given.')
            else:
                break

        arg_nodes = [None if ix is None else values[ix] for ix in value_ixs]
        # print('arg_nodes: {}'.format(['None' if n is None else n.value[0]
        #                               for n in arg_nodes]))
        arg_nodes += [None for _ in range(self.arity - len(arg_nodes))]
        return (op, arg_nodes)


class Agent:

    def __init__(self, env):
        self.env = env

    def train(self, episodes=5, alpha=0.1, gamma=0.6, epsilon=0.1):
        pass

        # env = self.env
        # qtable = self.qtable

        # # Run episodes
        # for _ in range(episodes):
        #     state = env.reset()
        #     epochs, penalties, reward, = 0, 0, 0

        #     while not env.state.done:
        #         pass

        #         if reward < 0:
        #             penalties += 1

        #         state = next_state
        #         epochs += 1

    def evaluate(self, episodes=5, empty_q_table=False):
        pass
