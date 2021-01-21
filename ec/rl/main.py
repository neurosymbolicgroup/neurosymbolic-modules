import sys
# should be running from ec directory!
sys.path.append(".")  # hack to make importing bidir work
from rl.environment import ArcEnvironment
from rl.create_ops import OP_DICT
from bidir.task_utils import get_task_examples
from rl.agent import ManualAgent, ArcAgent


def run_until_done(agent: ArcAgent, env: ArcEnvironment):
    """
    Basic sketch of running an agent on the environment.
    There may be ways of doing this which uses the gym framework better.
    Seems like there should be a way of modularizing the choice of RL
    algorithm, the agent choice, and the environment choice.
    """
    state = env.state
    done = env.done
    print('done: {}'.format(done))
    while not done:
        action = agent.choose_action(state)
        state, reward, done = env.step(action)
        print('reward: {}'.format(reward))


if __name__ == '__main__':
    ops = [
        'rotate_cw',
        'rotate_ccw',
        'vstack_pair',
        'vstack_pair_cond_inv',  # have to ask for inverse op explicitly
        'hflip',
        'hflip_inv',
        'vflip',
    ]
    ops = [OP_DICT[op_name] for op_name in ops]
    train_exs, test_exs = get_task_examples(115, train=True)
    env = ArcEnvironment(train_exs, test_exs, ops, max_actions=100)
    agent = ManualAgent(ops, env.arity, OP_DICT)

    run_until_done(agent, env)
