from rl.agent import RandomAgent
from rl.environment import SynthEnvAction
from rl.random_programs import ActionSpec, random_task
from rl.program_search_graph import ProgramSearchGraph, ValueNode
from bidir.task_utils import Task, twenty_four_task
from typing import List, Any, Tuple
import torch
import torch.optim as optim
import torch.cuda.amp as amp
from bidir.utils import assertEqual, num_params
import rl.ops.int_to_int_ops as int_ops
import rl.ops.twenty_four_ops as twenty_four_ops
from rl.policy_net import policy_net_int
from rl.environment import SynthEnv
import random
import time
import argparse
import itertools


def gcsl(
    net, env, steps, grad_freq=4, n_most_recent=None, batch_size=256, lr=5E-4, use_cuda=True, episode_print_every=10, step_print_every=100, use_amp=True,
):

    initial_random_actions = 10000
    print(f"number of parameters: {num_params(net)}")

    # for training
    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    max_arity = net.arg_choice_net.max_arity
    start = time.time()

    data_collection_time = 0
    training_time = 0
    data_start = None

    if use_cuda:
        net.cuda()
        criterion.cuda()
        if use_amp:
            scaler = amp.GradScaler()

    def supervised_update():

        def pad_list(lst, dim, pad_value=0):
            return list(lst) + [pad_value] * (dim - len(lst))

        net.train()
        optimizer.zero_grad()
        specs: List[ActionSpec] = [p.sample() for p in random.choices(buffer, k=batch_size)]
        psgs: List[ProgramSearchGraph] = [ProgramSearchGraph(spec.task, spec.additional_nodes) for spec in specs]
        op_classes = torch.tensor([s.action.op_idx for s in specs])
        args_classes = torch.stack([torch.tensor(pad_list(d.action.arg_idxs, max_arity)) for d in specs])

        with amp.autocast(enabled=use_amp):
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

        if use_amp:
            scaler.scale(combined_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
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
    solved_trajs: List[Tuple[ValueNode, List]] = []
    loss = 0
    current_steps = 0
    while current_steps <= steps:
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
                if episodes % (10 * episode_print_every) == 0:
                    for traj in solved_trajs:
                        print(f'\t{traj}')
                episodes_solved = []
                solved_trajs = []

            # print(env.actions_applied)
            # if no actions did anything, then nothing to use
            if len(obs.psg.get_grounded_nodes()) > len(obs.psg.inputs):
                buffer_point = BufferPoint(obs.psg, actions, n_nodes)
                buffer.append(buffer_point)
                if n_most_recent and len(buffer) >= n_most_recent:
                    buffer = buffer[-n_most_recent:]

            actions = []
            n_nodes = []
            obs = env.reset()

        if len(buffer) > batch_size:
            # don't count generation steps
            current_steps += 1
            train_start = time.time()
            if data_start:
                data_collection_time += train_start - data_start

            for i in range(grad_freq):
                loss += supervised_update()

            training_time += time.time() - train_start
            data_start = time.time()

            if current_steps % step_print_every == 0:
                print(f"loss = {loss} in time {time.time() - start}")
                print(f"total training time: {training_time}")
                print(f"total data col time: {data_collection_time}")
                print(f"percent spent training: {100 * training_time / (data_collection_time + training_time)}")
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


def int_training(use_amp=False, lr=5E-4, batch_size=256, big=False):
    ops = int_ops.FORWARD_OPS
    max_actions = 10
    use_cuda = torch.cuda.is_available()
    use_cuda = False
    int_ops.MAX_INT = 100
    max_int = int_ops.MAX_INT
    print(f"max_int: {max_int}")

    if big:
        deepset_layers=2
    else:
        deepset_layers=1

    net = policy_net_int(
        ops,
        max_int=max_int,
        state_dim=512,
        max_nodes=max_actions + 2,
        use_cuda=use_cuda,
        deepset_layers=deepset_layers)

    def task_sampler() -> Task:
        goal = random.choice(range(2, max_int + 1))
        return Task(((1, ), ), (goal, ))

    env = SynthEnv(ops, task_sampler=task_sampler, max_actions=max_actions)
    gcsl(net, env, steps=15000, grad_freq=4, use_cuda=use_cuda, lr=lr,
         batch_size=batch_size, episode_print_every=100, step_print_every=200,
         use_amp=use_amp)


def training_24(use_amp=False, lr=5E-4, batch_size=256, big=False):
    ops = twenty_four_ops.FORWARD_OPS
    max_actions = 5
    num_inputs = 3
    use_cuda = torch.cuda.is_available()
    max_int = 50
    max_input = 10
    twenty_four_ops.MAX_INT = max_int
    if big:
        deepset_layers=2
    else:
        deepset_layers=1
    # for now, sticking with binary embedding of numbers
    net = policy_net_int(
        ops,
        max_int=max_int,
        state_dim=512,
        max_nodes=max_actions + num_inputs + 1,
        use_cuda=use_cuda,
        deepset_layers=deepset_layers)

    def task_sampler() -> Task :
        inputs = random.sample(range(1, max_input + 1),
                               k=num_inputs)
        tuple_inputs = [(i, ) for i in inputs]
        task, _, _, _ = random_task(ops, tuple_inputs, depth=3)
        # goal = random.choice([a for a in range(2, max_int + 1) if a not in inputs])
        # return twenty_four_task(inputs, goal)
        return task

    env = SynthEnv(ops, task_sampler=task_sampler, max_actions=max_actions)
    gcsl(net, env, steps=15000, grad_freq=4, use_cuda=use_cuda, lr=lr,
         episode_print_every=100, step_print_every=200, use_amp=use_amp,
         batch_size=batch_size)

def job_number_to_hpo(job):
    amp_ops = [False]
    lr_ops = [5E-3, 5E-4, 5E-5]
    big = [True, False]
    batch_sizes = [256, 512]
    all_options = list(itertools.product(amp_ops, lr_ops, big, batch_sizes))
    option = all_options[job]
    return {'use_amp': option[0],
            'lr': option[1],
            'big': option[2],
            'batch_size': option[3]}


def random_hpo():
    # random number between 2 and 6
    rand1 = 2 + 4 * random.random()
    lr = 5 * 10**(-rand1)
    big = True if random.random() > 0.5 else False
    batch_size = random.choice([128, 256, 512])
    return {'use_amp': False,
            'lr': lr,
            'big': big,
            'batch_size': batch_size}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='gcsl')
    parser.add_argument("--use_amp",
                        action="store_true",
                        dest="use_amp")
    parser.add_argument("--lr",
                        type=float,
                        default=5E-4)
    parser.add_argument("--big",
                        action="store_true",
                        dest="big")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256)
    parser.add_argument("--use_job",
                        action="store_true",
                        dest="use_job")
    parser.add_argument("--job",
                        type=int,
                        default=0)
    parser.add_argument("--random",
                        action="store_true",
                        dest="random")

    args = parser.parse_args()
    if args.use_job:
        if args.random:
            d = random_hpo()
        else:
            d = job_number_to_hpo(args.job)
            print(f"job number = {args.job}")
        use_amp, lr, big, batch_size = d['use_amp'], d['lr'], d['big'], d['batch_size']
    else:
        use_amp, lr, big, batch_size = args.use_amp, args.lr, args.big, args.batch_size

    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(torch.cuda.device(0)))
    print(f"{use_amp=}, {lr=}, {big=}, {batch_size=}")
    int_training(use_amp=use_amp, lr=lr, big=big, batch_size=batch_size)
    # training_24(use_amp=use_amp, lr=lr, big=big, batch_size=batch_size)
