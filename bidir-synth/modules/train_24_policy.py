import time
import sys
import itertools
import random
import math
from modules.base_modules import FC
from rl.policy_net import policy_net_24
from rl.program_search_graph import ProgramSearchGraph
from bidir.utils import assertEqual, SynthError, next_unused_path
from bidir.task_utils import twenty_four_task
from rl.ops.operations import Op
import rl.ops.twenty_four_ops
from typing import Tuple, List, Dict, Any, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rl.random_programs import all_depth_one_programs


class DepthOneDataset(Dataset):
    def __init__(self, ops: Sequence[Op], num_inputs: int, max_input_int: int,
                 max_int: int):

        self.ops = ops
        self.num_inputs = num_inputs
        self.max_input_int = max_input_int
        self.max_int = max_int
        assert self.max_int >= self.max_input_int

        self.samples = all_depth_one_programs(self.ops, self.num_inputs,
                                              self.max_input_int, self.max_int)

        train_num = int(len(self.samples) * 0.8)
        self.held_out = self.samples[train_num:]
        self.samples = self.samples[:train_num]

        self.num_samples = len(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]

        def one_hot(i):
            return F.one_hot(torch.tensor(i), num_classes=self.max_int + 1)

        task, program = sample
        action = program[0]

        op_class = torch.tensor(action.op_idx)

        args_class = torch.tensor(action.arg_idxs)

        psg = ProgramSearchGraph(task)

        return {
            'sample': sample,
            'op_class': op_class,
            'args_class': args_class,
            'psg': psg,
        }


OP_DICT = {
    'a + b': lambda a, b: a + b,
    'a - b': lambda a, b: a - b,
    'a / b': lambda a, b: a // b,
    '2a - b': lambda a, b: 2 * a - b,
    'a - 2b': lambda a, b: a - 2 * b,
    '3a - b': lambda a, b: 3 * a - b,
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


class TwentyFourDataset2(Dataset):
    def __init__(self, num_ops: int, num_inputs: int, max_input_int: int,
                 max_int: int, num_samples: int):

        self.num_ops = num_ops
        self.num_inputs = num_inputs
        self.max_input_int = max_input_int
        self.max_int = max_int
        assert self.max_int >= self.max_input_int
        self.num_samples = num_samples

        self.op_dict = {
            op_str: op
            for (op_str, op) in list(OP_DICT.items())[0:self.num_ops]
        }
        self.op_str_to_idx = dict(
            zip(self.op_dict.keys(), list(range(len(self.op_dict)))))

        self.samples = self.generate_data()

    def generate_sample(self):
        good_choice = False
        attempts = 0
        while not good_choice:
            attempts += 1
            op_str, op = random.choice(list(self.op_dict.items()))
            inputs = random.sample(list(range(1, self.max_input_int + 1)),
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

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]

        def one_hot(i):
            return F.one_hot(torch.tensor(i), num_classes=self.max_int + 1)

        (a, b) = sample['args']
        out = sample['out']
        extras = sample['extras']
        op_str = sample['op_str']

        op_idx = self.op_str_to_idx[op_str]
        op_class = torch.tensor(op_idx)

        args = extras + [a, b]
        random.shuffle(args)
        a_idx = args.index(a)
        b_idx = args.index(b)

        args_class = torch.tensor([a_idx, b_idx])

        task = twenty_four_task(args, out)
        psg = ProgramSearchGraph(task)

        nodes = psg.get_value_nodes()
        ints = [n._value[0] for n in nodes]
        assertEqual(ints, args + [out])
        assert max(extras + [a, b] + [out]) <= self.max_int

        return {
            'sample': sample,
            'op_class': op_class,
            'args_class': args_class,
            'psg': psg,
        }


class TwentyFourDataset(Dataset):
    def __init__(self, ops: List[Op], transform=None):
        self.max_input_int = 16
        self.ops = ops
        self.op_names = [op.name for op in ops]
        self.op_dict = dict(zip(self.op_names, range(len(ops))))
        self.transform = transform
        self.samples = self.generate_data()
        self.max_int = max(max(a, b, out) for (a, b), out, _ in self.samples)
        self.in_dim = 3 * (self.max_int + 1)
        self.out_dim = len(self.ops)

    def generate_data2(self) -> List[Dict]:
        """
            ['args']: the two numbers
            ['extras']: extra numbers not used by op
            ['out']: the output number
            ['op_str']: the op string
        """
        pass

    def generate_data(self) -> List[Tuple[Tuple[int, int], int, str]]:
        inputs = [(i, j) for i in range(self.max_input_int + 1)
                  for j in range(self.max_input_int + 1)]

        num_unique = 0
        num_total = len(self.ops) * len(inputs)
        print('num_total: {}'.format(num_total))
        samples = []
        for ip in inputs:
            for op in self.ops:
                try:
                    out = op.forward_fn.fn(*ip)
                    if len(set([ip[0], ip[1], out])) < 3:
                        continue
                    samples.append((ip, out, op.name))
                except SynthError:
                    continue

        print(f"num samples: {len(samples)}")
        num_unique = len(set((ip, out) for (ip, out, name) in samples))
        print('num_unique: {}'.format(num_unique))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]
        if self.transform:
            sample = self.transform(sample)

        def one_hot(i):
            return F.one_hot(torch.tensor(i), num_classes=self.max_int + 1)

        (a, b), out, op_str = sample
        in_tens = torch.cat([one_hot(a), one_hot(b),
                             one_hot(out)]).to(torch.float32)
        op_idx = self.op_dict[op_str]
        op_class = torch.tensor(op_idx)
        args_class = torch.tensor([0, 1])

        task = twenty_four_task((a, b), out)
        psg = ProgramSearchGraph(task)

        nodes = psg.get_value_nodes()
        ints = [n._value[0] for n in nodes]
        assertEqual(ints, [a, b, out])

        return {
            'sample': sample,
            'in_tens': in_tens,
            'op_class': op_class,
            'args_class': args_class,
            'psg': psg
        }


def collate(batch):
    return {
        'sample': [el['sample'] for el in batch],
        'op_class': torch.stack([el['op_class'] for el in batch]),
        # (batch_size, arity)
        'args_class': torch.stack([el['args_class'] for el in batch]),
        'psg': [el['psg'] for el in batch],
    }


def train(net, data, epochs=100000, save_every=1000000, save_path=None):

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

                # print(batch['sample'])
                (ops, args), (op_logits, args_logits) = net(batch)
                # op_pred = None

                op_loss = criterion(op_logits, op_classes)

                # args_logits: (batch_size, n_classes, arity),
                # args_classes: (batch_size, arity)
                arg_loss = criterion(args_logits, args_classes)
                combined_loss = op_loss + arg_loss
                combined_loss.backward()
                optimizer.step()

                total_loss += combined_loss.sum().item()

                outs = [task.target[0] for (task, prog) in batch['sample']]
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

            print(
                f'Epoch {epoch} completed ({duration_str}) accuracy: {accuracy:.2f} loss: {total_loss:.2f}'
            )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), save_path)
        print('\nTraining interrupted with KeyboardInterrupt.',
              'Saved most recent model; exiting now.')
        sys.exit(0)

    print('Finished Training')


class FCNet(nn.Module):
    def __init__(self, in_dim, out_dim, ops: List[Op]):
        super().__init__()
        self.net = FC(input_dim=in_dim, num_hidden=2, output_dim=out_dim)
        self.ops = ops

    def forward(self, batch):
        in_tens = batch['in_tens']
        op_logits = torch.stack([self.net(in_ten) for in_ten in in_tens])
        op_idxs = torch.argmax(op_logits, dim=1)
        op_choices = [self.ops[i] for i in op_idxs]  # type: ignore

        return (op_choices, None), (op_logits, None)


class PolicyNetWrapper(nn.Module):
    def __init__(self, ops, max_int=None):
        super().__init__()
        self.ops = ops
        self.net = policy_net_24(ops, max_int=max_int, state_dim=512)

    def forward(self, batch):
        psgs = batch['psg']
        out = [self.net(psg) for psg in psgs]
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


def unwrap_wrapper_dict(state_dict):
    d = {}
    for key in state_dict:
        if key.startswith('net.'):
            d[key[4:]] = state_dict[key]

    return d


def main():
    data = DepthOneDataset(rl.ops.twenty_four_ops.FORWARD_OPS,
                           num_inputs=2,
                           max_input_int=5,
                           max_int=100)

    for i in range(min(10, data.num_samples)):
        print(data[i])
    print(f"Number of data points: {len(data)}")

    path = "depth=1_inputs=2_max_input_int=5.pt"
    ops = rl.ops.twenty_four_ops.FORWARD_OPS
    net = PolicyNetWrapper(ops, max_int=data.max_int)

    # net.load_state_dict(torch.load(path))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"number of parameters in model: {count_parameters(net)}")

    train(net, data, epochs=40000, save_every=100, save_path=path)
