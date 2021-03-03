import time
import sys
import math
import random
import itertools
from collections import namedtuple
from rl.policy_net import policy_net_24
from rl.program_search_graph import ProgramSearchGraph
from rl.environment import SynthEnvAction
from bidir.utils import assertEqual, SynthError, next_unused_path
from bidir.task_utils import twenty_four_task
from rl.ops.operations import Op
import rl.ops.twenty_four_ops
from typing import Tuple, List, Sequence
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rl.random_programs import depth_one_random_sample, DepthOneSpec


class DepthOneSampleDataset(Dataset):
    def __init__(
            self,
            ops: Sequence[Op],
            size: int,  # doesn't matter, just defines epoch size
            num_inputs: int,
            max_input_int: int,
            max_int: int = rl.ops.twenty_four_ops.MAX_INT,
            enforce_unique: bool = False):
        self.size = size
        self.ops = ops
        self.num_inputs = num_inputs
        self.max_input_int = max_input_int
        self.max_int = max_int
        self.enforce_unique = enforce_unique

    def __len__(self):
        """
        Note: since we're sampling randomly every time we get a new point, this
        doesn't really mean size. It is an arbitrary benchmark for the epoch.
        """
        return self.size

    def __getitem__(self, idx) -> DepthOneSpec:
        return depth_one_random_sample(self.ops, self.num_inputs,
                                       self.max_input_int, self.max_int,
                                       self.enforce_unique)


class TwentyFourDataset2(Dataset):
    def __init__(self, num_ops: int, num_inputs: int, max_input_int: int,
                 max_int: int, num_samples: int):

        self.num_ops = num_ops
        self.num_inputs = num_inputs
        self.max_input_int = max_input_int
        self.max_int = max_int
        assert self.max_int >= self.max_input_int
        self.num_samples = num_samples

        OP_DICT = {
            'a - b': lambda a, b: a - b,
            '2a - b': lambda a, b: 2 * a - b,
            '3a - b': lambda a, b: 3 * a - b,
            'a / b': lambda a, b: a // b,
            'a - 2b': lambda a, b: a - 2 * b,
            'a - 3b': lambda a, b: a - 3 * b,
            '3a - 2b': lambda a, b: 3 * a - 2 * b,
            '2a - 3b': lambda a, b: 2 * a - 3 * b,
            'a + 2b': lambda a, b: a + 2 * b,
            'a + 3b': lambda a, b: a + 3 * b,
            'a * (b + 1)': lambda a, b: a * (b + 1),
            'a * (b + 2)': lambda a, b: a * (b + 2),
            'a * (b + 3)': lambda a, b: a * (b + 3),
            '(a + 1) * (b + 2)': lambda a, b: (a + 1) * (b + 2),
            '(a + 1) * (b + 3)': lambda a, b: (a + 1) * (b + 3),
        }

        self.op_dict = {
            op_str: op
            for (op_str, op) in list(OP_DICT.items())[0:self.num_ops]
        }
        self.op_str_to_ix = dict(
            zip(self.op_dict.keys(), list(range(len(self.op_dict)))))

        self.samples = self.generate_data()

    def generate_sample(self):
        good_choice = False
        attempts = 0
        while not good_choice:
            attempts += 1
            op_str, op = random.choice(list(self.op_dict.items()))
            inputs = random.sample(list(range(1, self.max_input_int)),
                                   k=self.num_inputs)
            a, b = inputs[0:2]
            extras = inputs[2:]
            out = op(a, b)
            good_choice = float(out).is_integer(
            ) and out not in inputs and out > 0 and out < self.max_int
            if not good_choice:
                continue

            total_matches = 0
            for (c, d) in itertools.combinations(inputs, 2):
                matches = sum(op(c, d) == out for op in self.op_dict.values())
                total_matches += matches

            good_choice = total_matches == 1

        # otherwise, there are duplicate items, which means there should be
        # multiple matches
        assertEqual(len(list(set([a, b] + extras))), self.num_inputs)

        return {
            'args': (a, b),
            'extras': extras,
            'out': out,
            'op_str': op_str,
            'op': op,
            'attempts': attempts,
        }

    def generate_data(self):
        return [self.generate_sample() for _ in range(self.num_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> DepthOneSpec:
        sample = self.samples[idx]

        (a, b) = sample['args']
        out = sample['out']
        extras = sample['extras']
        op_str = sample['op_str']

        op_ix = self.op_str_to_ix[op_str]

        args = extras + [a, b]
        random.shuffle(args)
        a_ix = args.index(a)
        b_ix = args.index(b)

        task = twenty_four_task(args, out)
        action = SynthEnvAction(op_ix, (a_ix, b_ix))
        return DepthOneSpec(task, action)


def collate(batch):
    return {
        'sample': batch,
        'op_class': torch.tensor([d.action.op_idx for d in batch]),
        # (batch_size, arity)
        'args_class':
        torch.stack([torch.tensor(d.action.arg_idxs) for d in batch]),
        'psg': [ProgramSearchGraph(d.task) for d in batch],
    }


def train(net,
          data,
          epochs=100000,
          save_every=1000000,
          save_path=None,
          print_every=1,
          eval_every=100):

    dataloader = DataLoader(data, batch_size=128, collate_fn=collate)

    optimizer = optim.Adam(net.parameters(), lr=0.002)
    criterion = torch.nn.CrossEntropyLoss()

    if save_path:
        save_path = next_unused_path(save_path)

    try:  # if keyboard interrupt, will save net before exiting!
        for epoch in range(1, epochs + 1):
            start = time.time()
            total_loss = 0
            total_correct = 0

            if epoch > 0 and epoch % save_every == 0 and save_path:
                torch.save(net.state_dict(), save_path)

            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()

                op_classes = batch['op_class']
                # (batch_size, arity)
                args_classes = batch['args_class']

                (ops, args), (op_logits, args_logits) = net(batch, greedy=True)

                op_loss = criterion(op_logits, op_classes)

                # args_logits: (batch_size, n_classes, arity),
                # args_classes: (batch_size, arity)
                arg_loss = criterion(args_logits, args_classes)
                combined_loss = op_loss + arg_loss
                combined_loss.backward()
                optimizer.step()

                total_loss += combined_loss.sum().item()

                assert len(batch['sample'][0].task.target) == 1
                outs = [d.task.target[0] for d in batch['sample']]

                num_correct = 0

                for op, (a, b), out in zip(ops, args, outs):
                    try:
                        if op.forward_fn.fn(a, b) == out:
                            num_correct += 1
                    except SynthError:
                        pass
                total_correct += num_correct

            accuracy = 100 * total_correct / len(data)
            duration = time.time() - start
            m = math.floor(duration / 60)
            s = duration - m * 60
            duration_str = f'{m}m {int(s)}s'

            if epoch % print_every == 0:
                print(
                    f'Epoch {epoch} completed ({duration_str}) accuracy: {accuracy:.2f} loss: {total_loss:.2f}'
                )
            # if epoch % eval_every == 0:
            # accuracy = eval(net, data)
            # print(f"EVAL ACCURACY: {accuracy}")

    except KeyboardInterrupt:
        if save_path is not None:
            torch.save(net.state_dict(), save_path)
            print(
                "\nTraining interrupted with KeyboardInterrupt.",
                f"Saved most recent model at path {save_path}; exiting now.")
        print('Finished Training')
        sys.exit(0)

    print('Finished Training')


class PolicyNetWrapper(nn.Module):
    def __init__(self, ops, max_int=None, state_dim=512):
        super().__init__()
        self.ops = ops
        self.net = policy_net_24(ops, max_int=max_int, state_dim=state_dim)

    def forward(self, batch, greedy: bool = False):
        psgs = batch['psg']
        out = [self.net(psg, greedy=greedy) for psg in psgs]
        ops = []
        arg_choices: List[Tuple[int, int]] = []
        op_logits = []
        arg_logits = []

        for (op_idx, arg_idxs, op_logit, arg_logit), psg in zip(out, psgs):
            nodes = psg.get_value_nodes()
            assert len(arg_idxs) == 2
            arg1, arg2 = nodes[arg_idxs[0]], nodes[arg_idxs[1]]
            ops.append(self.net.ops[op_idx])
            arg_choices.append((arg1.value[0], arg2.value[0]))
            op_logits.append(op_logit)
            arg_logit = torch.transpose(arg_logit, 0, 1)
            assertEqual(arg_logit.shape[1], self.net.arg_choice_net.max_arity)
            arg_logits.append(arg_logit)

        op_logits2 = torch.stack(op_logits)
        arg_logits2 = torch.stack(arg_logits)
        return (ops, arg_choices), (op_logits2, arg_logits2)


def eval(net, data) -> int:
    net.eval()

    dataloader = DataLoader(data, batch_size=256, collate_fn=collate)

    total_correct = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            (ops, args), (op_logits, args_logits) = net(batch, greedy=True)

            assert len(batch['sample'][0].task.target) == 1
            outs = [d.task.target[0] for d in batch['sample']]

            num_correct = 0

            for op, (a, b), out in zip(ops, args, outs):
                try:
                    if op.forward_fn.fn(a, b) == out:
                        num_correct += 1
                except SynthError:
                    pass
            total_correct += num_correct

        accuracy = 100 * total_correct / len(data)
        net.train()
        return accuracy


def unwrap_wrapper_dict(state_dict):
    d = {}
    for key in state_dict:
        if key.startswith('net.'):
            d[key[4:]] = state_dict[key]

    return d


def main():

    # ops = rl.ops.twenty_four_ops.FORWARD_OPS
    num_ops = 5
    ops = rl.ops.twenty_four_ops.SPECIAL_FORWARD_OPS[0:num_ops]
    data = DepthOneSampleDataset(ops=ops,
                                 size=1000,
                                 num_inputs=3,
                                 max_input_int=10,
                                 enforce_unique=True)

    # data = TwentyFourDataset2(num_ops=5,
    #                           num_inputs=5,
    #                           max_input_int=16,
    #                           max_int=100,
    #                           num_samples=1000)

    # Op = namedtuple('Op', ['name', 'arity', 'forward_fn'])
    # ops = [Op(s, 2, namedtuple('Function', ['fn'])(f)) for (s, f) in data.op_dict.items()]

    for i in range(min(20, len(data))):
        print(data[i])
    print(f"Number of data points: {len(data)}")

    old_path = "models/net_3_3_2.pt"
    path = "models/net_3_3_3.pt"
    net = PolicyNetWrapper(ops, max_int=data.max_int)

    net.load_state_dict(torch.load(old_path))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"number of parameters in model: {count_parameters(net)}")

    train(net, data, epochs=2000, print_every=1, save_path=path, save_every=100)
