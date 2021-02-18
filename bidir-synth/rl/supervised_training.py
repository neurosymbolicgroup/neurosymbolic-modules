import time
import sys
import math
from rl.policy_net import policy_net_24
from rl.program_search_graph import ProgramSearchGraph
from bidir.utils import assertEqual, SynthError, next_unused_path
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
        return self.size

    def __getitem__(self, idx) -> DepthOneSpec:
        return depth_one_random_sample(self.ops, self.num_inputs,
                                       self.max_input_int, self.max_int,
                                       self.enforce_unique)


def collate(batch):
    return {
        'sample': batch,
        'op_class': torch.tensor([d.action.op_idx for d in batch]),
        # (batch_size, arity)
        'args_class': torch.stack([torch.tensor(d.action.arg_idxs)
                                   for d in batch]),
        'psg': [ProgramSearchGraph(d.task) for d in batch],
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

                (ops, args), (op_logits, args_logits) = net(batch)

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

            print(
                f'Epoch {epoch} completed ({duration_str}) accuracy: {accuracy:.2f} loss: {total_loss:.2f}'
            )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), save_path)
        print("\nTraining interrupted with KeyboardInterrupt.",
              f"Saved most recent model at path {save_path}; exiting now.")
        sys.exit(0)

    print('Finished Training')


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
    data = DepthOneSampleDataset(ops=rl.ops.twenty_four_ops.FORWARD_OPS,
                                 size=10,
                                 num_inputs=2,
                                 max_input_int=5,
                                 enforce_unique=True)

    for i in range(min(100, len(data))):
        print(data[i])
    print(f"Number of data points: {len(data)}")

    path = "models/depth=1_inputs=2_max_input_int=5.pt"
    ops = rl.ops.twenty_four_ops.FORWARD_OPS
    net = PolicyNetWrapper(ops, max_int=data.max_int)

    # net.load_state_dict(torch.load(path))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"number of parameters in model: {count_parameters(net)}")

    train(net, data, epochs=40000)  #, save_every=100, save_path=path)