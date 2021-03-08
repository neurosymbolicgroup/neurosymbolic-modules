"""
Performs supervised training on the op-chooser subnet for depth 1 tasks.
"""

import math
import random
import sys
import time
from typing import List, Sequence

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from bidir.utils import next_unused_path
from rl.policy_net import policy_net_24, PolicyPred
from rl.program_search_graph import ProgramSearchGraph
from rl.random_programs import depth_one_random_24_sample, DepthOneSpec
from rl.ops.operations import Op
import rl.ops.twenty_four_ops


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
        return depth_one_random_24_sample(self.ops, self.num_inputs,
                                       self.max_input_int, self.max_int,
                                       self.enforce_unique)


def collate(batch: List[DepthOneSpec]):
    return {
        "op_idx": torch.tensor([d.action.op_idx for d in batch]),
        # (batch_size, arity)
        "arg_idxs":
        torch.stack([torch.tensor(d.action.arg_idxs) for d in batch]),
        "psg": [ProgramSearchGraph(d.task) for d in batch],
    }


def train(
    net,
    data,
    epochs=100000,
    save_every=1000000,
    save_path=None,
):
    dataloader = DataLoader(data, batch_size=128, collate_fn=collate)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    loss_criterion = torch.nn.CrossEntropyLoss()

    if save_path:
        save_path = next_unused_path(save_path)

    try:  # if keyboard interrupt, will save net before exiting!
        for epoch in range(1, epochs + 1):
            start = time.time()
            total_loss = 0
            total_correct = 0

            per_op_totals = [0 for _ in net.ops]
            per_op_correct = [0 for _ in net.ops]

            if epoch > 0 and epoch % save_every == 0 and save_path:
                torch.save(net.state_dict(), save_path)

            for batch in dataloader:
                batch_preds: List[PolicyPred] = [
                    net(psg) for psg in batch["psg"]
                ]

                op_loss = loss_criterion(
                    torch.stack([pred.op_logits for pred in batch_preds]),
                    batch["op_idx"],
                )

                # arg_loss = loss_criterion(
                #     torch.stack([
                #         torch.transpose(pred.arg_logits, 0, 1)
                #         for pred in batch_preds
                #     ]),
                #     batch["arg_idxs"],
                # )

                combined_loss = op_loss  # + arg_loss
                total_loss += combined_loss

                optimizer.zero_grad()
                combined_loss.backward()
                optimizer.step()

                op_correct = torch.tensor(
                    [pred.op_idx for pred in batch_preds]) == batch["op_idx"]

                # args_correct = torch.all(
                #     torch.stack([
                #         torch.tensor(pred.arg_idxs) for pred in batch_preds
                #     ]) == batch["arg_idxs"],
                #     dim=1,s
                # )

                num_correct = op_correct.sum()  # & args_correct).sum()
                total_correct += num_correct

                for i in range(len(net.ops)):
                    mask = (batch["op_idx"] == i)
                    per_op_totals[i] += mask.sum()
                    per_op_correct[i] += op_correct[mask].sum()

            accuracy = 100 * total_correct / len(data)
            duration = time.time() - start
            m = math.floor(duration / 60)
            s = duration - m * 60
            duration_str = f'{m}m {int(s)}s'

            print(
                f"Epoch {epoch} completed ({duration_str}) accuracy: {accuracy:.3f} loss: {total_loss:.3f}"
            )
            for i in range(len(net.ops)):
                print(
                    f"{net.ops[i].name}; acc: {per_op_correct[i] / per_op_totals[i]:.3f}; weight: {per_op_totals[i] / len(data):.3f}"
                )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), save_path)
        print("\nTraining interrupted with KeyboardInterrupt.",
              f"Saved most recent model at path {save_path}; exiting now.")
        sys.exit(0)

    print("Finished Training")


def main():
    seed = 47
    random.seed(seed)
    torch.manual_seed(seed)

    ops = rl.ops.twenty_four_ops.FORWARD_OPS
    data = DepthOneSampleDataset(
        ops=ops,
        size=1024,
        num_inputs=3,
        max_input_int=24,
        enforce_unique=True,
    )

    for _ in range(10):
        print(data[0])
    print(f"Number of data points: {len(data)}")

    net = policy_net_24(
        ops,
        max_int=rl.ops.twenty_four_ops.MAX_INT,
        state_dim=512,
    )

    # net.load_state_dict(torch.load(path))

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"number of parameters in model: {count_parameters(net)}")

    train(
        net,
        data,
        epochs=40000,
        # save_path="models/depth=1.pt&inputs=2&max_input=10.pt",
        # save_every=100,
    )


if __name__ == "__main__":
    main()
