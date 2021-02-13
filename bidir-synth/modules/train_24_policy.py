import time
import itertools
import random
import math
from modules.base_modules import FC
from rl.policy_net import PolicyNet24
from rl.program_search_graph import ProgramSearchGraph
from bidir.utils import assertEqual, SynthError
from rl.operations import Op
from typing import List, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
# from bidir.twenty_four import OP_DICT

OP_DICT = {
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

    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.samples[idx]

        def one_hot(i):
            return F.one_hot(torch.tensor(i), num_classes=self.max_int + 1)

        (a, b) = sample['args']
        out = sample['out']
        extras = sample['extras']
        op_str = sample['op_str']

        op_ix = self.op_str_to_ix[op_str]
        op_class = torch.tensor(op_ix)

        args = extras + [a, b]
        random.shuffle(args)
        a_ix = args.index(a)
        b_ix = args.index(b)

        args_class = torch.tensor([a_ix, b_ix])

        start_values = tuple((i, ) for i in args)
        end_value = (out, )
        psg = ProgramSearchGraph(start_values, end_value)

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
        op_ix = self.op_dict[op_str]
        op_class = torch.tensor(op_ix)
        args_class = torch.tensor([0, 1])

        start_values = ((a, ), (b, ))
        end_value = (out, )
        psg = ProgramSearchGraph(start_values, end_value)

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


def train(net, data, epochs=100000):

    dataloader = DataLoader(data, batch_size=128, collate_fn=collate)

    optimizer = optim.Adam(net.parameters(), lr=0.002)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        start = time.time()
        total_loss = 0
        total_correct = 0

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            op_classes = batch['op_class']
            # (batch_size, arity)
            args_classes = batch['args_class']

            # print(batch['sample'])
            (op_pred, args_pred), (op_logits, args_logits) = net(batch)
            # op_pred = None

            op_loss = criterion(op_logits, op_classes)

            # args_logits: (batch_size, n_classes, arity),
            # args_classes: (batch_size, arity)
            arg_loss = criterion(args_logits, args_classes)
            combined_loss = op_loss + arg_loss
            combined_loss.backward()
            optimizer.step()

            total_loss += combined_loss.sum().item()

            if args_pred is None:
                assert False
                args_pred = [sample['args'] for sample in batch['sample']]

            if op_pred is None:
                assert False
                op_pred = [sample['op'] for sample in batch['sample']]

            outs = [sample['out'] for sample in batch['sample']]
            num_correct = 0

            for op, (a, b), out in zip(op_pred, args_pred, outs):
                try:
                    if op.fn(a, b) == out:
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

    print('Finished Training')


class FCNet(nn.Module):
    def __init__(self, in_dim, out_dim, ops: List[Op]):
        super().__init__()
        self.net = FC(input_dim=in_dim, num_hidden=2, output_dim=out_dim)
        self.ops = ops

    def forward(self, batch):
        in_tens = batch['in_tens']
        op_logits = torch.stack([self.net(in_ten) for in_ten in in_tens])
        op_ixs = torch.argmax(op_logits, dim=1)
        op_choices = [self.ops[i] for i in op_ixs]

        return (op_choices, None), (op_logits, None)


class PointerNet(nn.Module):
    def __init__(self, ops, node_dim=None):
        super().__init__()
        self.ops = ops
        self.net = PolicyNet24(ops, node_dim=node_dim, state_dim=512)

    def forward(self, batch):
        psgs = batch['psg']
        out = [self.net(psg) for psg in psgs]
        op_choices = []
        arg_choices: List[Tuple[int, int]] = []
        op_logits = []
        arg_logits = []
        for (op_chosen, args), (op_logit, arg_logit) in out:
            op_choices.append(op_chosen)
            assert len(args) == 2
            arg_choices.append((args[0].value[0], args[1].value[0]))
            op_logits.append(op_logit)
            arg_logits.append(arg_logit)

        op_logits2 = torch.stack(op_logits)
        arg_logits2 = torch.stack(arg_logits)
        return (op_choices, arg_choices), (op_logits2, arg_logits2)


def main_old():
    op_strs = ['sub', 'div']
    ops = [OP_DICT[o] for o in op_strs]
    # list of ((a, b), out, op_str)
    data = TwentyFourDataset(ops)

    print(f"Number of data points: {len(data)}")

    # net1 = FCNet(data.in_dim, data.out_dim, ops)
    net2 = PointerNet(ops, node_dim=data.max_int + 1)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"number of parameters in model: {count_parameters(net2)}")
    # FC net becomes too large if we increase the inputs too much
    # train(net1, data)
    train(net2, data, epochs=400)


def main():
    data = TwentyFourDataset2(num_ops=5,
                              num_inputs=5,
                              max_input_int=16,
                              max_int=100,
                              num_samples=1000)

    for i in range(10):
        print(data[i])
    print(f"Number of data points: {len(data)}")

    # PolicyNet24 should work with strings for ops as well
    Op = namedtuple('Op', ['name', 'arity', 'fn'])
    ops = [Op(s, 2, OP_DICT[s]) for s in data.op_dict.keys()]
    net = PointerNet(ops, node_dim=data.max_int + 1)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"number of parameters in model: {count_parameters(net)}")

    train(net, data, epochs=400)
