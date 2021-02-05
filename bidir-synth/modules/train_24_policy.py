import time
import math
from modules.base_modules import FC
from rl.policy_net import PolicyNet24
from rl.program_search_graph import ProgramSearchGraph
from bidir.utils import assertEqual
from rl.operations import Op
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from bidir.twenty_four import OP_DICT, TwentyFourError


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

    def generate_data(self):
        inputs = [(i, j) for i in range(self.max_input_int + 1)
                  for j in range(self.max_input_int + 1)]

        num_unique = 0
        num_total = len(self.ops) * len(inputs)
        print('num_total: {}'.format(num_total))
        samples = []
        for ip in inputs:
            for op in self.ops:
                try:
                    out = op.fn.fn(*ip)
                    if len(set([ip[0], ip[1], out])) < 3:
                        continue
                    samples.append((ip, out, op.name))
                except TwentyFourError:
                    continue

        print(f"num samples: {len(samples)}")
        num_unique = len(set((ip, out) for (ip, out, name) in samples))
        print('num_unique: {}'.format(num_unique))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[Tuple[int, int], int, str]:
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
            # 'args_class': args_class,
            'psg': psg
        }


def collate(batch):
    return {
        'sample': [el['sample'] for el in batch],
        'in_tens': torch.stack([el['in_tens'] for el in batch]),
        'op_class': torch.stack([el['op_class'] for el in batch]),
        'psg': [el['psg'] for el in batch],
    }


def train(net, data, epochs=100000):

    dataloader = DataLoader(data, batch_size=128, collate_fn=collate)

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        start = time.time()
        total_loss = 0
        total_correct = 0

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()

            op_classes = batch['op_class']

            # print(batch['sample'])
            (op_pred, args_pred), (op_logits, args_logits) = net(batch)

            loss = criterion(op_logits, op_classes)
            loss.backward()
            optimizer.step()

            total_loss += loss.sum().item()

            num_correct = 0
            for op, ((a, b), out, _) in zip(op_pred, batch['sample']):
                try:
                    if op.fn.fn(a, b) == out:
                        num_correct += 1
                    # else:
                        # if epoch > 20:
                            # print((a, b), out, op.name)
                except Exception:
                    pass
            total_correct += num_correct

        accuracy = 100 * total_correct / len(data)
        duration = time.time() - start
        m = math.floor(duration / 60)
        s = duration - m * 60
        duration_str = f'{m}m {int(s)}s'

        print(
            f'Epoch {epoch} completed ({duration_str}) accuracy: {accuracy:.2f} loss: {loss:.2f}'
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
        self.net = PolicyNet24(ops, node_dim=node_dim)

    def forward(self, batch):
        psgs = batch['psg']
        out = [self.net(psg) for psg in psgs]
        op_choices = []
        arg_choices = []
        op_logits = []
        arg_logits = []
        for (op_chosen, args), (op_logit, arg_logit) in out:
            op_choices.append(op_chosen)
            arg_choices.append(args)
            op_logits.append(op_logit)
            arg_logits.append(arg_logit)

        op_logits = torch.stack(op_logits)
        # op_ixs = torch.argmax(op_logits, dim=1)
        # op_choices = [self.ops[i] for i in op_ixs]
        # arg_logits = torch.stack(arg_logits)
        arg_logits = None
        return (op_choices, arg_choices), (op_logits, arg_logits)


def main():
    op_strs = ['add', 'sub', 'mul', 'div']
    ops = [OP_DICT[o] for o in op_strs]
    # list of ((a, b), out, op_str)
    data = TwentyFourDataset(ops)

    print(f"Number of data points: {len(data)}")

    # net1 = FCNet(data.in_dim, data.out_dim, ops)
    net2 = PointerNet(ops, data.max_int + 1)
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print(count_parameters(net1))
    print(f"number of parameters in model: {count_parameters(net2)}")
    # FC net becomes too large if we increase the inputs too much
    # train(net1, data)
    # should get to 100% accuracy after around 180 epochs
    train(net2, data, epochs=200)
