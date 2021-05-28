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

from rl.agent_program import rl_prog_solves
from rl.environment import SynthEnvAction
from rl.program_search_graph import ProgramSearchGraph, ValueNode
import rl.ops.twenty_four_ops
import rl.ops.arc_ops
from rl.random_programs import ActionSpec, ProgramSpec, random_bidir_program
import bidir.utils as utils
from rl.random_programs import random_arc_small_grid_inputs_sampler
from rl.policy_net import policy_net_arc, policy_net_24
from rl.test_search import policy_rollouts


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
        spec = utils.load_action_spec(self.directory + '/' +
                                      self.file_names[idx])
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
        size: int = 1000,
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


def collate(batch: Sequence[ActionSpec], max_arity: int = 2):
    def pad_list(lst, dim, pad_value=0):
        return list(lst) + [pad_value] * (dim - len(lst))

    return {
        'sample': batch,
        'op_class': torch.tensor([d.action.op_idx for d in batch]),
        # (batch_size, arity)
        # 'args_class':
        # torch.stack([torch.tensor(d.action.arg_idxs) for d in batch]),
        'args_class': torch.stack([torch.tensor(pad_list(d.action.arg_idxs, max_arity)) for d in batch]),
        'psg': [ProgramSearchGraph(d.task, d.additional_nodes) for d in batch],
    }


def train(
    net,
    data,  # __getitem__ should return an ActionSpec
    val_data=None,
    lr=0.002,
    # batch_size=5000,
    batch_size=128,
    epochs=300,
    print_every=1,
    use_cuda=True,
    save_every=0,
    rollout_fn=None,
    test_every=0,
):

    max_arity = net.arg_choice_net.max_arity
    dataloader = DataLoader(data,
                            batch_size=batch_size,
                            collate_fn=lambda batch: collate(batch, max_arity))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    # best_val_accuracy = -1

    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    metrics: Dict[str, Any] = {}
    try:  # if keyboard interrupt, will save net before exiting!
        for epoch in range(epochs):
            if test_every and epoch % test_every == 0 and rollout_fn:
                solved_tasks, attempts_per_task = rollout_fn()
                mlflow.log_metrics({'epoch': epoch,
                                    'solved': len(solved_tasks),
                                    'attempted_to_solve': len(attempts_per_task)})
                print(f"solved {len(solved_tasks)} tasks out of {len(attempts_per_task)}")

            start = time.time()
            total_loss = 0
            op_correct = 0.
            args_correct = 0.
            exact_correct = 0.

            net.train()

            last_task = None
            for batch_num, batch in enumerate(dataloader):
                last_task = batch['psg'][-1].task

                optimizer.zero_grad()
                # last batch might be smaller than batch_size
                this_batch_size = len(batch['psg'])

                op_classes = batch['op_class']
                # (batch_size, arity)
                args_classes = batch['args_class']

                op_idxs, args_idxs, op_logits, args_logits = net(batch['psg'],
                                                                 greedy=True)
                # print(f"args_idxs: {args_idxs}")
                # print(f"args_classes: {args_classes}")

                op_correct += (op_classes == op_idxs).sum().item()
                args_correct += ((args_classes == args_idxs).sum().item()
                                 / net.arg_choice_net.max_arity)

                corrects = [(op_idx == op_class).item()
                            and all(args_idx == args_class)
                            for (op_idx, op_class, args_idx, args_class)
                            in zip(op_idxs, op_classes,
                                   args_idxs, args_classes)]

                assert all([isinstance(c, bool) for c in corrects])

                exact_correct += sum(corrects)

                # print(f"corrects: {corrects}")
                # print(sum(corrects))

                if False:
                    if not sum(corrects):
                        continue
                    op_logits = torch.stack([a for (a, b) in zip(op_logits, corrects) if b])
                    op_classes = torch.stack([a for (a, b) in zip(op_classes, corrects) if b])
                    args_logits = torch.stack([a for (a, b) in zip(args_logits, corrects) if b])
                    args_classes = torch.stack([a for (a, b) in zip(args_classes, corrects) if b])

                if use_cuda:
                    op_logits = op_logits.cuda()
                    op_classes = op_classes.cuda()
                    args_logits = args_logits.cuda()
                    args_classes = args_classes.cuda()

                # if batch_num == 0:
                #     print(f"op_idxs: {op_idxs}")
                #     print(f"op_classes: {op_classes}")
                #     print(f"op_logits: {op_logits}")
                op_loss = criterion(op_logits, op_classes)

                # N = len(batch['psg'])
                max_arity = net.arg_choice_net.max_arity
                nodes = net.max_nodes

                # utils.assertEqual(args_classes.shape, (N, max_arity))
                # utils.assertEqual(args_logits.shape, (N, max_arity, nodes))

                args_logits = args_logits.permute(0, 2, 1)
                # utils.assertEqual(args_logits.shape, (N, nodes, max_arity))

                arg_loss = criterion(args_logits, args_classes)

                combined_loss = op_loss + arg_loss
                combined_loss.backward()

                optimizer.step()

                total_loss += combined_loss.sum().item()

            # print(f"last_task: {last_task}")

            op_accuracy = 100 * op_correct / len(data)
            args_accuracy = 100 * args_correct / len(data)
            # solves the task the same way as the training example did
            exact_accuracy = 100 * exact_correct / len(data)

            duration = time.time() - start
            m = math.floor(duration / 60)
            s = duration - m * 60
            duration_str = f'{m}m {s:.1f}s'

            if save_every and epoch > 0 and epoch % save_every == 0:
                utils.save_mlflow_model(net,
                                        model_name=f"epoch-{metrics['epoch']}")

            if epoch % print_every == 0:
                print(
                    f'Epoch {epoch} completed ({duration_str})',
                    f'op_accuracy: {op_accuracy:.2f}',
                    f'args_accuracy: {args_accuracy:.2f}',
                    f'loss: {total_loss:.2f}',
                    f'exact_accuracy: {exact_accuracy:.2f}',
                )

            metrics = dict(
                epoch=epoch,
                op_accuracy=op_accuracy,
                args_accuracy=args_accuracy,
                exact_accuracy=exact_accuracy,
                loss=total_loss,
            )

            mlflow.log_metrics(metrics, step=epoch)

        return op_accuracy, args_accuracy
    except KeyboardInterrupt:
        pass

    # save when done, or if we interrupt.
    if save_every:
        utils.save_mlflow_model(net)


def twenty_four_batched_train():
    data_size = 1000
    depth = 1
    num_inputs = 2
    max_input_int = 15
    max_int = rl.ops.twenty_four_ops.MAX_INT

    ops = rl.ops.twenty_four_ops.SPECIAL_FORWARD_OPS[0:5]

    def sampler():
        inputs = random.sample(range(1, max_input_int + 1), k=num_inputs)
        tuple_inputs = tuple((i, ) for i in inputs)
        return random_bidir_program(ops, tuple_inputs, depth, forward_only=True)

    programs = [sampler() for _ in range(data_size)]
    data = program_dataset(programs)

    use_cuda = False
    use_cuda = use_cuda and torch.cuda.is_available()  # type: ignore

    net = policy_net_24(ops, max_int=max_int, state_dim=128, use_cuda=use_cuda)
    train(net, data, epochs=300, print_every=5, use_cuda=use_cuda)


def arc_batched_train():
    data_size = 1000
    depth = 1

    ops = rl.ops.arc_ops.FW_GRID_OPS

    random_arc_small_grid_inputs_sampler

    def sampler():
        inputs = random_arc_small_grid_inputs_sampler()
        prog: ProgramSpec = random_bidir_program(ops,
                                                 inputs,
                                                 depth=depth,
                                                 forward_only=True)
        return prog

    programs = [sampler() for _ in range(data_size)]
    data = program_dataset(programs)
    # for i in range(len(data)):
    #     d = data.data[i]
    #     print(d.task, d.additional_nodes, d.action)

    use_cuda = False
    use_cuda = use_cuda and torch.cuda.is_available()  # type: ignore
    net = policy_net_arc(ops, state_dim=5, use_cuda=use_cuda)
    train(net, data, epochs=100, print_every=1, use_cuda=use_cuda)


if __name__ == '__main__':
    random.seed(44)
    torch.manual_seed(44)
    arc_batched_train()
    # twenty_four_batched_train()
