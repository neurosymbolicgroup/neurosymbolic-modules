import os
import time
import math
import random

from typing import List, Sequence, Callable, Dict, Any

import mlflow

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from rl.program_search_graph import ProgramSearchGraph, ValueNode
import rl.ops.twenty_four_ops
from rl.random_programs import ActionSpec, random_24_program, ProgramSpec, random_bidir_program
from bidir.utils import save_mlflow_model, load_action_spec
from rl.random_programs import random_arc_small_grid_inputs_sampler
from rl.policy_net import policy_net_arc, policy_net_24


class ActionDatasetOnDisk2(Dataset):
    """
    This just combines a bunch into one dataset.
    """
    def __init__(
        self,
        dirs: Sequence[str],
    ):
        super().__init__()
        self.dirs = dirs
        self.sub_datasets = [ActionDatasetOnDisk(dir) for dir in self.dirs]

    def __len__(self):
        """
        Arbitrary length just to set epoch size.
        """
        return 1000

    def __getitem__(self, idx) -> ActionSpec:
        return self.random_sample()

    def random_sample(self) -> ActionSpec:
        i = random.choice(range(len(self.dirs)))
        return self.sub_datasets[i].random_sample()


class ActionDatasetOnDisk(Dataset):
    def __init__(
        self,
        directory: str,
    ):
        """
        If fixed_set is true, then samples size points and only trains on
        those. Otherwise, calls the sampler anew when training.

        fixed_set = True should be used if you have a really large space of
        possible samples, and only want to train on a subset of them.
        """
        super().__init__()
        self.directory = directory
        self.file_names = os.listdir(self.directory)
        self.size = len(self.file_names)

    def __len__(self):
        """
        Arbitrary length just to set epoch size.
        """
        return 1000

    def __getitem__(self, idx) -> ActionSpec:
        return self.random_sample()

    def random_sample(self) -> ActionSpec:
        idx = random.choice(range(self.size))
        spec = load_action_spec(self.directory + '/' + self.file_names[idx])
        return spec




class ActionDataset(Dataset):
    def __init__(
        self,
        data: List[ActionSpec],
    ):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> ActionSpec:
        return self.data[idx]


class ActionSamplerDataset(Dataset):
    def __init__(
        self,
        sampler: Callable[[], ActionSpec],
        size: int,
    ):
        super().__init__()
        self.size = size
        self.sampler = sampler

    def __len__(self):
        """
        This is an arbitrary size set for the epoch.
        """
        return self.size

    def __getitem__(self, idx) -> ActionSpec:
        return self.sampler()


def program_dataset(program_specs: Sequence[ProgramSpec]) -> ActionDataset:
    action_specs: List[ActionSpec] = []
    for program_spec in program_specs:
        action_specs += program_spec.action_specs

    return ActionDataset(action_specs)


class NodeEmbedNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, node: ValueNode, is_grounded: bool) -> Tensor:
        raise NotImplementedError


def collate(batch: Sequence[ActionSpec],
            max_arity: int = 2):
    def pad_list(lst, dim, pad_value=0):
        return list(lst) + [pad_value]*(dim - len(lst))

    return {
        'sample': batch,
        'op_class': torch.tensor([d.action.op_idx for d in batch]),
        # (batch_size, arity)
        # 'args_class':
        # torch.stack([torch.tensor(d.action.arg_idxs) for d in batch]),
        'args_class': [torch.tensor(pad_list(d.action.arg_idxs, max_arity)) for d in batch],
        'psg': [ProgramSearchGraph(d.task, d.additional_nodes) for d in batch],
    }


def train(net,
          node_embed_net,
          data,  # __getitem__ should return an ActionSpec
          val_data=None,
          lr=0.002,
          batch_size=128,
          epochs=300,
          print_every=1,
          use_cuda=True,
          max_nodes=100,
          save_model=True,
          save_every=-1):

    mlflow.log_params({'lr': lr})

    max_arity = net.arg_choice_net.max_arity
    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            collate_fn=lambda batch: collate(batch, max_arity))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    print('warning: node embed optimizer disabled')
    # node_embed_optimizer = optim.Adam(node_embed_net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    # best_val_accuracy = -1

    use_cuda = use_cuda and torch.cuda.is_available()  # type: ignore

    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    metrics: Dict[str, Any]  = {}
    try:  # if keyboard interrupt, will save net before exiting!
        for epoch in range(epochs):
            if (save_model and save_every > 0 and epoch > 0
                    and epoch % save_every == 0):
                save_mlflow_model(
                    net,
                    model_name=f"epoch-{metrics['epoch']}")

            start = time.time()
            total_loss = 0
            op_accuracy = 0.
            args_accuracy = 0.
            iters = 0

            net.train()
            # node_embed_net.train()

            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()
                # node_embed_optimizer.zero_grad()

                op_classes = batch['op_class']
                # (batch_size, arity)
                args_classes = batch['args_class']

                batch_tensor = torch.zeros(op_classes.shape[0], max_nodes, node_embed_net.dim)
                for j, psg in enumerate(batch['psg']):
                    assert len(psg.get_value_nodes()) <= max_nodes
                    for k, node in enumerate(psg.get_value_nodes()):
                        batch_tensor[j, k, :] = node_embed_net(node, psg.is_grounded(node))

                if use_cuda:
                    batch_tensor = batch_tensor.cuda()
                    op_classes = op_classes.cuda()

                op_idxs, arg_idxs, op_logits, args_logits = net(batch_tensor, greedy=True)
                # if i == 0:
                #     print(f"op_idxs: {op_idxs}")
                #     print(f"op_classes: {op_classes}")
                #     print(f"op_logits: {op_logits}")
                op_loss = criterion(op_logits, op_classes)

                args_classes_tensor = torch.stack(args_classes).cuda() if use_cuda else torch.stack(args_classes)
                arg_losses = [criterion(args_logits[:, i, :],
                                        args_classes_tensor[:, i])
                              for i in range(net.arg_choice_net.max_arity)]

                arg_loss = sum(arg_losses) / len(arg_losses)

                combined_loss = op_loss + arg_loss
                combined_loss.backward()

                optimizer.step()
                # node_embed_optimizer.step()

                total_loss += combined_loss.sum().item()

                op_accuracy += (op_classes == op_idxs).float().mean().item()*100
                args_accuracy += (args_classes_tensor == arg_idxs).float().mean().item()*100
                iters += 1

            op_accuracy = op_accuracy/iters
            args_accuracy = args_accuracy/iters
            # val_accuracy = eval_policy_net(net, data)
            # if val_accuracy > best_val_accuracy:
            #     best_val_accuracy, best_epoch = val_accuracy, epoch
            duration = time.time() - start
            m = math.floor(duration / 60)
            s = duration - m * 60
            duration_str = f'{m}m {int(s)}s'

            if epoch % print_every == 0:
                print(
                    f'Epoch {epoch} completed ({duration_str})',
                    f'op_accuracy: {op_accuracy:.2f}',
                    f'args_accuracy: {args_accuracy:.2f} loss: {total_loss:.2f}',
                )

            metrics = dict(
                epoch=epoch,
                op_accuracy=op_accuracy,
                args_accuracy=args_accuracy,
                loss=total_loss,
            )

            mlflow.log_metrics(metrics, step=epoch)

    except KeyboardInterrupt:
        pass

    # save when done, or if we interrupt.
    if save_model:
        save_mlflow_model(net)
    return op_accuracy, args_accuracy  # , best_val_accuracy


def arc_training():
    data_size = 100
    val_data_size = 200
    depth = 1

    ops = rl.ops.arc_ops.FW_GRID_OPS

    def sampler():
        inputs = random_arc_small_grid_inputs_sampler()
        prog: ProgramSpec = random_bidir_program(ops,
                                                 inputs,
                                                 depth=depth,
                                                 forward_only=True)
        return prog

    programs = [sampler() for _ in range(data_size)]
    data = program_dataset(programs)
    # for i in range(3):
    #     d = data.data[i]
    #     print(d.task, d.additional_nodes, d.action)

    val_programs = [sampler() for _ in range(val_data_size)]
    val_data = program_dataset(val_programs)

    net, node_embed_net = policy_net_arc(ops, state_dim=128)
    op_accuracy, args_accuracy = train_policy_net(net, node_embed_net, data, val_data, print_every=1, use_cuda=True, epochs=1000, batch_size=data_size, lr=0.001, max_nodes=2)
    print(op_accuracy, args_accuracy)


def twenty_four_batched_test():
    data_size = 1000
    val_data_size = 200
    depth = 1
    num_inputs = 2
    max_input_int = 15
    max_int = rl.ops.twenty_four_ops.MAX_INT

    ops = rl.ops.twenty_four_ops.SPECIAL_FORWARD_OPS[0:5]

    def sampler():
        inputs = random.sample(range(1, max_input_int + 1), k=num_inputs)
        return random_24_program(ops, inputs, depth)

    programs = [sampler() for _ in range(data_size)]
    data = program_dataset(programs)

    val_programs = [sampler() for _ in range(val_data_size)]
    val_data = program_dataset(val_programs)

    net, node_embed_net = policy_net_24(ops, max_int=max_int, state_dim=128)
    op_accuracy, args_accuracy = train_policy_net(net, node_embed_net, data, val_data, print_every=5, use_cuda=True)
    print(op_accuracy, args_accuracy)


if __name__ == '__main__':
    # arc_training()
    twenty_four_batched_test()
