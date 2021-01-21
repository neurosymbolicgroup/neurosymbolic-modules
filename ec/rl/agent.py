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
            print(f'{i}:\t({type(val.value)})\t{str(val)}')

        print("Choose an op (provided as string, e.g. 'vstack_pair_cond_inv')")
        op = input('Choice: ')
        op = self.op_dict[op]

        print('Args for op, as index of value list printed. If cond. inverse',
              ' provide output then inputs, with masks for unknown inputs')
        print("e.g. '1, None, 2' for vstack_pair_cond_inv")
        value_ixs = input('Choice: ')
        value_ixs = value_ixs.strip()
        value_ixs = value_ixs.split(',')
        value_ixs = [int(ix) for ix in value_ixs]

        arg_nodes = [None if ix is None else values[ix] for ix in value_ixs]
        print('arg_nodes: {}'.format([n.value for n in arg_nodes]))
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
