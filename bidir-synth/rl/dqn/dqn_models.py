from typing import List, Tuple, NamedTuple, Optional, Sequence

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from bidir.utils import assertEqual

from modules.base_modules import FC
from modules.synth_modules import DeepSetNet

from rl.environment import SynthEnvAction, SynthEnvObservation
from rl.ops.operations import Op
from rl.random_programs import random_action
from rl.policy_net import NodeEmbedNet
from rl.program_search_graph import ValueNode


class SynthQValues(NamedTuple):
    op_qvals: Tensor  # shape: (num_ops,)
    op_idx: int

    # arg_qvals are "conditioned" on having picked op_idx
    arg_qvals: Tensor  # shape (arity, num_input_nodes)
    greedy_arg_idxs: Tuple[int, ...]


def get_relevant_qvals(
    q: SynthQValues,
    arg_idxs: Tuple[int, ...],
) -> Tensor:
    assertEqual(len(arg_idxs), len(q.greedy_arg_idxs))

    op_qval = q.op_qvals[q.op_idx]
    arg_qvals = [q.arg_qvals[i, arg_idx] for i, arg_idx in enumerate(arg_idxs)]

    return torch.stack([op_qval] + arg_qvals)


class ArgQNetDirectChoice(nn.Module):
    def __init__(
        self,
        ops: Sequence[Op],
        node_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.ops = ops
        self.num_ops = len(ops)
        self.max_arity = max(op.arity for op in ops)

        self.node_dim = node_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.net = FC(input_dim=self.state_dim + self.num_ops + self.node_dim,
                      output_dim=self.max_arity,
                      num_hidden=1,
                      hidden_dim=self.hidden_dim)

    def forward(
        self,
        op_idx: int,
        node_embeddings: Tensor,
        state_embed: Tensor,
    ) -> Tuple[Tensor, Tuple[int, ...]]:
        """
        Equivalent to a pointer net, but chooses args all at once.
        Much easier to understand, too.
        """
        op_arity = self.ops[op_idx].arity
        num_nodes = node_embeddings.shape[0]

        op_one_hot = F.one_hot(torch.tensor(op_idx), num_classes=self.num_ops)

        assertEqual(node_embeddings.shape, (num_nodes, self.node_dim))
        assertEqual(op_one_hot.shape, (self.num_ops, ))
        assertEqual(state_embed.shape, (self.state_dim, ))

        # in tensor: (N, state_dim + op_dim + node_dim)
        query = torch.cat([op_one_hot, state_embed])
        query = query.repeat(num_nodes, 1)
        in_tensor = torch.cat([query, node_embeddings], dim=1)
        assertEqual(
            in_tensor.shape,
            (num_nodes, self.num_ops + self.state_dim + self.node_dim),
        )

        # process each node if separate elements in a batch
        arg_qvals_all = self.net(in_tensor).T
        arg_qvals = arg_qvals_all[:op_arity]
        assertEqual(arg_qvals.shape, (op_arity, num_nodes))

        greedy_arg_idxs_t: Tensor = torch.argmax(arg_qvals, dim=1)
        greedy_arg_idxs: Tuple[int, ...] = tuple(greedy_arg_idxs_t.tolist())
        assertEqual(len(greedy_arg_idxs), op_arity)

        return arg_qvals, greedy_arg_idxs


# TODO: better networks.


class SynthQNet(nn.Module):
    def __init__(
        self,
        ops: Sequence[Op],
        node_embed_net: NodeEmbedNet,
        hidden_dim: int,
    ):
        super().__init__()

        self.ops = ops
        self.num_ops = len(ops)
        self.hidden_dim = hidden_dim

        self.node_embed_net = node_embed_net

        # for embedding the ProgramSearchGraph
        self.deepset_net = DeepSetNet(
            element_dim=self.node_embed_net.dim,
            hidden_dim=self.hidden_dim,
            presum_num_layers=1,
            postsum_num_layers=1,
            set_dim=self.hidden_dim,
        )

        # for op_qvals
        self.op_linear = FC(
            input_dim=self.hidden_dim,
            output_dim=self.num_ops,
            num_hidden=1,
            hidden_dim=self.num_ops,
        )

        # for arg_qvals
        self.arg_qnet = ArgQNetDirectChoice(
            ops=self.ops,
            node_dim=self.node_embed_net.dim,
            state_dim=self.hidden_dim,
        )

    def forward(
        self,
        obs: SynthEnvObservation,
        forced_op_idx: Optional[int] = None,
    ) -> SynthQValues:
        nodes: List[ValueNode] = obs.psg.get_value_nodes()
        node_embeddings: Tensor = torch.stack([
            self.node_embed_net(node, obs.psg.is_grounded(node))
            for node in nodes
        ])

        psg_embedding: Tensor = self.deepset_net(node_embeddings)
        assertEqual(psg_embedding.shape, (self.hidden_dim, ))

        op_qvals: Tensor = self.op_linear(psg_embedding)
        if forced_op_idx is None:
            op_idx = torch.argmax(op_qvals).item()
        else:
            op_idx = forced_op_idx
        assert isinstance(op_idx, int)  # for type-checking

        arg_qvals, greedy_arg_idxs = self.arg_qnet.forward(
            op_idx=op_idx,
            node_embeddings=node_embeddings,
            state_embed=psg_embedding,
        )

        return SynthQValues(
            op_qvals=op_qvals,
            op_idx=op_idx,
            arg_qvals=arg_qvals,
            greedy_arg_idxs=greedy_arg_idxs,
        )


def epsilon_greedy_sample(
    sqn: SynthQNet,
    eps: float,
    obs: SynthEnvObservation,
    no_grad: bool = False,
) -> SynthEnvAction:
    # With eps probability, take a random action
    if np.random.rand() < eps:
        return random_action(sqn.ops, obs.psg)

    # Otherwise greedily sample using the sqn

    if no_grad:
        with torch.no_grad():
            qvals = sqn.forward(obs=obs)
    else:
        qvals = sqn.forward(obs=obs)

    arg_idxs_t: Tensor = torch.argmax(qvals.arg_qvals, dim=1)
    arg_idxs: Tuple[int, ...] = tuple(arg_idxs_t.tolist())

    return SynthEnvAction(op_idx=qvals.op_idx, arg_idxs=arg_idxs)
