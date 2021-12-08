from rl.agent import RandomAgent
from rl.environment import SynthEnvAction
from rl.random_programs import ActionSpec
from rl.program_search_graph import ProgramSearchGraph, ValueNode
from bidir.task_utils import Task
from typing import List, Any, Tuple
import torch
import torch.optim as optim
from bidir.utils import assertEqual
import rl.ops.int_to_int_ops as int_ops
from rl.policy_net import policy_net_int
from rl.environment import SynthEnv
import random
import time


def gcsl(
    net, env, steps, grad_freq=4, n_most_recent=None, batch_size=256, lr=5E-4, use_cuda=True, episode_print_every=10, step_print_every=100
):

    initial_random_actions = 10000

    # for training
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    max_arity = net.arg_choice_net.max_arity
    start = time.time()

    if use_cuda:
        net.cuda()
        criterion.cuda()

    def supervised_update():

        def pad_list(lst, dim, pad_value=0):
            return list(lst) + [pad_value] * (dim - len(lst))

        net.train()
        optimizer.zero_grad()
        specs: List[ActionSpec] = [p.sample() for p in random.choices(buffer, k=batch_size)]
        psgs: List[ProgramSearchGraph] = [ProgramSearchGraph(spec.task, spec.additional_nodes) for spec in specs]
        op_classes = torch.tensor([s.action.op_idx for s in specs])
        args_classes = torch.stack([torch.tensor(pad_list(d.action.arg_idxs, max_arity)) for d in specs])

        op_idxs, args_idxs, op_logits, args_logits = net(psgs, greedy=True)

        if use_cuda:
            op_logits = op_logits.cuda()
            op_classes = op_classes.cuda()
            args_logits = args_logits.cuda()
            args_classes = args_classes.cuda()

        op_loss = criterion(op_logits, op_classes)

        nodes = net.max_nodes
        assertEqual(args_classes.shape, (batch_size, max_arity))
        assertEqual(args_logits.shape, (batch_size, max_arity, nodes))

        args_logits = args_logits.permute(0, 2, 1)
        assertEqual(args_logits.shape, (batch_size, nodes, max_arity))

        arg_loss = criterion(args_logits, args_classes)

        combined_loss = op_loss + arg_loss
        combined_loss.backward()

        optimizer.step()

        return combined_loss.item()

    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    obs = env.reset()
    random_agent = RandomAgent(net.ops)
    actions: List[SynthEnvAction] = []
    n_nodes = []

    buffer = []

    episodes = 0
    episodes_solved = []
    solved_trajs = []
    loss = 0
    current_steps = 0
    while current_steps <= steps:
        current_steps += 1
        net.eval()
        with torch.no_grad():
            preds = net([obs.psg], greedy=True)

        if current_steps <= initial_random_actions:
            act = random_agent.choose_action(obs)
        else:
            act = SynthEnvAction(preds.op_idxs[0].item(),
                                 preds.arg_idxs[0].tolist())
        actions.append(act)
        n_nodes.append(len(obs.psg.get_grounded_nodes()))
        obs, rew, done, _ = env.step(act)

        if done:
            episodes += 1
            solved = obs.psg.solved()
            episodes_solved.append(solved)
            if solved and len(solved_trajs) < 5:
                solved_trajs.append((env.task.target[0], env.actions_applied))

            if episodes % episode_print_every == 0:
                percent_solved = sum(episodes_solved) / len(episodes_solved)
                print(f"{percent_solved}% of last {episode_print_every} episodes solved")
                if episodes % 10 * episode_print_every == 0:
                    for traj in solved_trajs:
                        print(f'\t{traj}')
                episodes_solved = []
                solved_trajs = []

            # print(env.actions_applied)
            # if no actions did anything, then nothing to use
            if len(obs.psg.get_grounded_nodes()) > 2:
                buffer_point = BufferPoint(obs.psg, actions, n_nodes)
                buffer.append(buffer_point)
                if n_most_recent and len(buffer) >= n_most_recent:
                    buffer = buffer[-n_most_recent:]

            actions = []
            n_nodes = []
            obs = env.reset()

        if len(buffer) > batch_size:
            for i in range(grad_freq):
                loss += supervised_update()

            if current_steps % step_print_every == 0:
                print(f"loss = {loss} in time {time.time() - start}")
                loss = 0
                start = time.time()



class BufferPoint():
    def __init__(self, psg, actions, n_nodes):
        self.actions = actions
        self.n_nodes = n_nodes
        self.length = len(actions)
        self.nodes: List[ValueNode] = psg.get_grounded_nodes()
        # print(f"nodes: {self.nodes}, len={len(self.nodes)}")
        # print(f"n_nodes: {n_nodes}")
        self.n_goal_options = [len(self.nodes) - n_nodes for n_nodes in self.n_nodes]
        self.input_node = self.nodes[0]

    def sample(self) -> ActionSpec:
        step = random.choices(range(self.length), self.n_goal_options)[0]
        # the last n_goal_options elements are possible
        goal_ix = len(self.nodes) - 1 - random.choices(range(self.n_goal_options[step]))[0]

        input_node: ValueNode = self.nodes[0]
        inputs: Tuple[Tuple[Any, ...], ...] = (input_node.value, )
        # needs is_grounded attribute
        additional_nodes = [(n, True) for n in self.nodes[1:self.n_nodes[step]]]
        goal: Tuple[Any, ...] = self.nodes[goal_ix].value
        assert goal not in [n for (n, b) in additional_nodes]
        assert goal != input_node.value
        task = Task(inputs, goal)
        action_spec = ActionSpec(task, self.actions[step], additional_nodes)
        return action_spec


if __name__ == '__main__':
    ops = int_ops.FORWARD_OPS
    max_actions = 10
    use_cuda = torch.cuda.is_available()
    int_ops.MAX_INT = 100
    max_int = int_ops.MAX_INT
    net = policy_net_int(
        ops,
        max_int=max_int,
        state_dim=512,
        max_nodes=max_actions + 2,
        use_cuda=use_cuda)

    def task_sampler() -> Task :
        goal = random.choice(range(2, max_int + 1))
        return Task(((1, ), ), (goal, ))

    env = SynthEnv(ops, task_sampler=task_sampler, max_actions=max_actions)
    gcsl(net, env, steps=5E10, grad_freq=4, use_cuda=use_cuda, lr=5E-4,
            episode_print_every=100, step_print_every=1000)
