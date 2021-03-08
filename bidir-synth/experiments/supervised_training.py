import time
import math
import mlflow
import random
from typing import Tuple, List, Sequence, Dict, Any, Callable

import torch
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from bidir.utils import assertEqual, SynthError, load_mlflow_model, save_mlflow_model, USE_CUDA
from rl.policy_net import policy_net_24
from rl.program_search_graph import ProgramSearchGraph
import rl.ops.twenty_four_ops
from rl.ops.operations import ForwardOp, Op
from rl.random_programs import ActionSpec, depth_one_random_24_sample, random_24_program, ProgramSpec


class ActionDataset(Dataset):
    def __init__(
        self,
        sampler: Callable[[], ActionSpec],
        size: int,
        fixed_set: bool = False,
    ):
        """
        If fixed_set is true, then samples size points and only trains on
        those. Otherwise, calls the sampler anew when training.

        fixed_set = True should be used if you have a really large space of
        possible samples, and only want to train on a subset of them.
        """
        super().__init__()
        self.size = size
        self.sampler = sampler
        self.fixed_set = fixed_set

        if fixed_set:
            self.data = [sampler() for _ in range(size)]

    def __len__(self):
        """
        If fixed_set is false, this is an arbitrary size set for the epoch.
        """
        return self.size

    def __getitem__(self, idx) -> ActionSpec:
        if self.fixed_set:
            return self.data[idx]
        else:
            return self.sampler()


def program_dataset(program_specs: Sequence[ProgramSpec]) -> ActionDataset:
    action_specs: List[ActionSpec] = []
    for program_spec in program_specs:
        action_specs += program_spec.action_specs

    def sampler():
        return random.choice(action_specs)

    return ActionDataset(sampler, size=len(action_specs), fixed_set=False)


def collate(batch: Sequence[ActionSpec]):
    return {
        'sample': batch,
        'op_class': torch.tensor([d.action.op_idx for d in batch]),
        # (batch_size, arity)
        'args_class':
        torch.stack([torch.tensor(d.action.arg_idxs) for d in batch]),
        'psg': [ProgramSearchGraph(d.task) for d in batch],
    }


def batch_inference(
    net,
    batch: Dict[str, Any],
    greedy: bool = True
) -> Tuple[Tuple[List[Op], List[Tuple[int, int]]], Tuple[Tensor,
                                                         List[Tensor]]]:
    """
    the policy net only takes one psg at a time, but we want to evaluate a
    batch of psgs.
    """
    psgs = batch['psg']
    out = [net(psg, greedy=greedy) for psg in psgs]
    ops = []
    arg_choices: List[Tuple[Any, ...]] = []
    op_logits = []
    arg_logits = []
    max_num_logits = 0

    for (op_idx, arg_idxs, op_logit, arg_logit), psg in zip(out, psgs):
        nodes = psg.get_value_nodes()

        ops.append(net.ops[op_idx])
        op_logits.append(op_logit)

        # don't feel like implementing multi-example checking atm
        assert all(len(nodes[arg_idx].value) == 1 for arg_idx in arg_idxs)
        args = tuple(nodes[arg_idx].value[0] for arg_idx in arg_idxs)

        arg_choices.append(args)

        arg_logit = torch.transpose(arg_logit, 0, 1)
        assertEqual(arg_logit.shape[1], net.arg_choice_net.max_arity)
        # keep arg logits in a list, instead of stacking, because the choices
        # may have been made from different numbers of input args, in which
        # case the cross entropy loss has to be done for each choice
        # separately.
        arg_logits.append(arg_logit)

    op_logits2 = torch.stack(op_logits)
    return (ops, arg_choices), (op_logits2, arg_logits)


def train(
    net,
    data,  # __getitem__ should return a ActionSpec
    epochs=1e100,
    save_model=False,
    print_every=1,
    lr=0.002,
    batch_size=128,
):

    TRAIN_PARAMS = dict(
        batch_size=128,
        lr=0.002,
    )

    mlflow.log_params(TRAIN_PARAMS)

    dataloader = DataLoader(data, batch_size=batch_size, collate_fn=collate)

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    if USE_CUDA:
        criterion = criterion.cuda()

    try:  # if keyboard interrupt, will save net before exiting!
        for epoch in range(1, epochs + 1):
            start = time.time()
            total_loss = 0
            total_correct = 0

            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()

                op_classes = batch['op_class']
                # (batch_size, arity)
                args_classes = batch['args_class']
                if USE_CUDA:
                    op_classes, args_classes = op_classes.cuda(), args_classes.cuda()

                (ops, args), (op_logits,
                              args_logits) = batch_inference(net,
                                                             batch,
                                                             greedy=True)

                op_loss = criterion(op_logits, op_classes)

                # args_logits: (batch_size, n_classes, arity),
                # args_classes: (batch_size, arity)
                # arg_loss = criterion(args_logits, args_classes)
                # args_logits: List of tensors (n_classes, arity)
                # switched to list since sometimes n_classes differs between
                # examples
                # one day, we might want to optimize all of this...
                arg_losses = [
                        # criterion expects batches, so unsqueeze a batch dim
                    criterion(torch.unsqueeze(arg_logit, dim=0),
                              torch.unsqueeze(arg_class, dim=0))
                    for arg_logit, arg_class in zip(args_logits, args_classes)
                ]
                arg_loss = sum(arg_losses) / len(arg_losses)

                combined_loss = op_loss + arg_loss
                combined_loss.backward()
                optimizer.step()

                total_loss += combined_loss.sum().item()

                # only checks the first example for accuracy
                assert len(batch['sample'][0].task.target) == 1
                outs = [d.task.target[0] for d in batch['sample']]

                num_correct = 0

                for op, inputs, out in zip(ops, args, outs):
                    # only works with ForwardOps for now
                    assert isinstance(op, ForwardOp)
                    try:
                        if op.forward_fn.fn(*inputs) == out:
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
                    f'Epoch {epoch} completed ({duration_str})',
                    f'accuracy: {accuracy:.2f} loss: {total_loss:.2f}',
                )

            metrics = dict(
                epoch=epoch,
                accuracy=accuracy,
                loss=total_loss,
            )

            mlflow.log_metrics(metrics)

    except KeyboardInterrupt:
        pass

    # save when done, or if we interrupt.
    if save_model:
        save_mlflow_model(net)


def eval(net, data) -> float:
    """
    data is a Dataset whose __getitem__ returns a ActionSpec
    """
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
    """
    In case the model was an old PolicyNetWrapper nn.Module
    """
    d = {}
    for key in state_dict:
        if key.startswith('net.'):
            d[key[4:]] = state_dict[key]

    return d


def main():

    mlflow.set_experiment("Supervised training")
    data_size = 1000
    depth = 1
    num_inputs = 2
    max_input_int = 5
    max_int = rl.ops.twenty_four_ops.MAX_INT
    enforce_unique = False
    # model_load_run_id = "de47b2faec3c4ccb90a7e4f4f09beccf"
    model_load_run_id = None
    save_model = True

    with mlflow.start_run():
        PARAMS = dict(
            data_size=data_size,
            num_inputs=num_inputs,
            max_input_int=max_input_int,
            max_int=max_int,
            enforce_unique=enforce_unique,
            model_load_run_id=model_load_run_id,
            save_model=save_model,
        )

        mlflow.log_params(PARAMS)

        # ops = rl.ops.twenty_four_ops.FORWARD_OPS[0:1]
        ops = rl.ops.twenty_four_ops.SPECIAL_FORWARD_OPS[0:5]
        print(f"ops: {ops}")

        def spec_sampler():
            return depth_one_random_24_sample(ops,
                                           num_inputs=num_inputs,
                                           max_input_int=max_input_int,
                                           max_int=max_int,
                                           enforce_unique=enforce_unique)

        # data = ActionDataset(
        #     size=data_size,
        #     sampler=spec_sampler,
        #     fixed_set=False,
        # )
        def sampler():
            inputs = random.sample(range(1, max_input_int + 1), k=num_inputs)
            return random_24_program(ops, inputs, depth)

        programs = [sampler() for _ in range(data_size)]
        data = program_dataset(programs)

        print('Preview of data points:')
        for i in range(min(10, len(data))):
            print(data.__getitem__(i))  # simply calling data[i] doesn't work

        if model_load_run_id:
            net = load_mlflow_model(model_load_run_id)
        else:
            net = policy_net_24(ops, max_int=max_int, state_dim=512)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

        print(f"number of parameters in model: {count_parameters(net)}")

        print(f"Starting run:\n{mlflow.active_run().info.run_id}")
        train(
            net,
            data,
            epochs=300,
            print_every=1,
            save_model=save_model,
        )
