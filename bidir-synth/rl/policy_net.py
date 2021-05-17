from typing import List, Tuple, NamedTuple, Sequence
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.categorical import Categorical

from bidir.primitives.types import Grid, MIN_COLOR, NUM_COLORS
from bidir.utils import assertEqual

import modules.synth_modules
from modules.synth_modules import PointerNet2
from modules.base_modules import FC
from rl.ops.operations import Op
from rl.program_search_graph import ValueNode, ProgramSearchGraph


def compute_out_size(in_size, model: nn.Module):
    """
    Compute output size of model for an input with size `in_size`.
    This is a utility function.
    """
    out = model(torch.Tensor(1, *in_size))  # type: ignore
    return int(np.prod(out.size()))


class PolicyPred(NamedTuple):
    op_idxs: Tensor
    arg_idxs: Tensor
    op_logits: Tensor
    arg_logits: Tensor  # shape (max_arity, num_input_nodes)


class ArcNodeEmbedNet(nn.Module):
    # TODO: batchize?
    def __init__(self, dim):
        super().__init__()
        self.aux_dim = 1  # extra dim to encode groundedness
        self.embed_dim = dim - self.aux_dim

        # TODO: will have to turn grid numpy array into torch tensor with
        # different channel for each color
        # self.CNN = CNN(in_channels=len(Color), output_dim=self.node_dim)

        # takes in sequence of embeddings, and produces an embedding of same
        # dimensionality.
        # self.LSTM = LSTM(input_dim=self.node_dim,
        #                  hidden_dim=64,
        #                  output_dim=self.node_dim)

    def forward(self, node: ValueNode, is_grounded: bool) -> Tensor:
        """
            Embeds the node to dimension self.D (= self.type_embed_dim +
            self.node_aux_dim)
            - first embeds the value by type to dimension self.type_embed_dim
            - then concatenates auxilliary node dimensions of dimension
              self.node_aux_dim
        """
        # embedding example list same way we would embed any other list
        examples: Tuple = node.value

        # 0 or 1 depending on whether node is grounded or not.
        grounded_tensor = torch.tensor([int(is_grounded)])

        out = torch.cat([self.embed_by_type(examples), grounded_tensor])
        return out

    def embed_by_type(self, value):
        # possible types: Tuples/Lists, Grids, Arrays, bool/int/color.
        if isinstance(value, tuple) or isinstance(value, list):
            subembeddings = [self.embed_by_type(x) for x in value]
            return self.embed_tensor_list(subembeddings)
        elif isinstance(value, Grid):
            return self.embed_grid(value)
        # elif isinstance(value, Array):
        # raise NotImplementedError
        elif isinstance(value, int):
            # TODO: make a linear layer for these
            raise NotImplementedError

    def embed_tensor_list(self, emb_list) -> Tensor:
        raise NotImplementedError
        # TODO: format input for LSTM
        out = self.LSTM(emb_list)
        return out

    def embed_grid(self, grid: Grid) -> Tensor:
        raise NotImplementedError
        # TODO: convert grid into correct input
        # Maybe we want a separate method for embedding batches of grids?
        batched_tensor = grid
        out = self.CNN(batched_tensor)
        return out


class TwentyFourNodeEmbedNet(nn.Module):
    def __init__(self, max_int):
        super().__init__()
        self.aux_dim = 1  # extra dim to encode groundedness
        self.embed_dim = max_int + 1
        self.dim = self.embed_dim + self.aux_dim

    def forward(self, node: ValueNode, is_grounded: bool) -> Tensor:
        assert isinstance(node.value, tuple)
        assertEqual(len(node.value), 1)
        n = node.value[0]
        assert isinstance(n, int)

        # values range from zero to MAX_INT inclusive
        out = F.one_hot(torch.tensor(n), num_classes=self.embed_dim)
        out = out.to(torch.float32)

        out = torch.cat([out, torch.tensor([int(is_grounded)])])
        assertEqual(out.shape, (self.dim, ))
        return out


class ArcNodeEmbedNetGridsOnly(nn.Module):
    """Only embeds nodes with values of type Tuple[Grid]"""
    def __init__(self, dim, side_len=32, use_cuda=True):
        """
        dim is the output dimension
        side_len is what grids get padded/cropped to internally
        """
        super().__init__()
        self.dim = dim
        self.aux_dim = 1  # extra dim to encode groundedness
        self.embed_dim = dim - self.aux_dim
        self.use_cuda = use_cuda

        self.side_len = side_len
        self.num_channels = NUM_COLORS + 1

        # We use 2D convolutions because there is only spatial structure
        # (hence need for convolution locality) in the height and width
        # dimensions.
        # Number of intermediate channels chosen to keep number of outputs
        # at each intermediate stage constant.
        # TODO: Tune architecture
        self.CNN = nn.Sequential(
            nn.Conv2d(in_channels=self.num_channels,
                      out_channels=32,
                      kernel_size=4,
                      stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                      stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=4,
                      stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.CNN_output_size = compute_out_size(
            in_size=(self.num_channels, self.side_len, self.side_len),
            model=self.CNN,
        )
        # For debugging:
        # print(self.CNN_output_size)
        # assert False

        self.embedding_combiner = modules.synth_modules.DeepSetNet(
            element_dim=self.CNN_output_size,
            hidden_dim=self.embed_dim,
            presum_num_layers=1,
            postsum_num_layers=1,
            set_dim=self.embed_dim,
        )

        if self.use_cuda:
            self.cuda()

    def forward(self, node: ValueNode, is_grounded: bool) -> Tensor:
        """
            Embeds the node to dimension self.D (= self.type_embed_dim +
            self.node_aux_dim)
            - first embeds the value by type to dimension self.type_embed_dim
            - then concatenates auxilliary node dimensions of dimension
              self.node_aux_dim
        """
        grids: Tuple[Grid, ...] = node.value
        assert isinstance(grids[0], Grid)
        # TODO: undo limitation to one example only
        grids = grids[0:1]

        t_grids = self.grids_to_tensor(grids)
        grid_embeddings = self.CNN(t_grids)

        combined_embeddings = self.embedding_combiner(grid_embeddings)

        grounded_tens = torch.tensor([int(is_grounded)])
        if self.use_cuda:
            grounded_tens = grounded_tens.cuda()

        out = torch.cat([
            combined_embeddings,
            grounded_tens,
        ])

        return out

    def grids_to_tensor(self, grids: Tuple[Grid, ...]) -> Tensor:
        assert len(grids) > 0

        def resize(a: np.ndarray):
            """resize by cropping or padding with zeros"""
            od1, od2 = a.shape
            rd1, rd2 = min(od1, self.side_len), min(od2, self.side_len)

            ret = np.zeros((self.side_len, self.side_len))
            ret[:rd1, :rd2] = a[:rd1, :rd2]

            return ret

        np_arrs = [g.arr for g in grids]
        np_arrs_shifted = [a - MIN_COLOR + 1 for a in np_arrs]

        t_padded = torch.tensor([resize(a) for a in np_arrs_shifted],
                                dtype=torch.int64)
        if self.use_cuda:
            t_padded = t_padded.cuda()

        t_onehot = F.one_hot(t_padded, num_classes=self.num_channels)
        assertEqual(
            t_onehot.shape,
            (len(grids), self.side_len, self.side_len, self.num_channels),
        )

        ret = t_onehot.permute(0, 3, 1, 2)
        assertEqual(
            ret.shape,
            (len(grids), self.num_channels, self.side_len, self.side_len),
        )

        return ret.to(torch.float32)


class DirectChoiceNet(nn.Module):
    def __init__(self, ops: Sequence[Op], node_dim: int, state_dim: int):
        super().__init__()
        self.ops = ops
        self.node_dim = node_dim
        self.state_dim = state_dim
        self.num_ops = len(ops)
        self.max_arity = max(op.arity for op in ops)

        self.args_net = FC(input_dim=self.state_dim + self.num_ops +
                           self.node_dim,
                           output_dim=self.max_arity,
                           num_hidden=1,
                           hidden_dim=256)

    def forward(
        self,
        op_idxs: Tensor,
        state_embeds: Tensor,
        psg_embeddings: Tensor,
        greedy: bool = False,
    ) -> Tuple[Tensor, Tensor]:
        """
        Equivalent to a pointer net, but chooses args all at once.
        Much easier to understand, too.
        """
        N = psg_embeddings.shape[0]
        num_nodes = psg_embeddings.shape[1]

        op_one_hot = F.one_hot(op_idxs, num_classes=self.num_ops)
        assertEqual(psg_embeddings.shape, (N, num_nodes, self.node_dim))
        assertEqual(op_one_hot.shape, (N, self.num_ops))
        assertEqual(state_embeds.shape, (N, self.state_dim))

        query = torch.cat([op_one_hot, state_embeds], dim=1)
        query = query.unsqueeze(1)
        query = query.repeat(1, num_nodes, 1)
        # query[psg_embeddings == 0.] = 0.
        in_tensor = torch.cat([query, psg_embeddings], dim=2)
        assertEqual(in_tensor.shape,
                    (N, num_nodes, self.state_dim + self.num_ops + self.node_dim))

        # process each node if separate elements in a batch
        arg_logits = self.args_net(in_tensor)
        assertEqual(arg_logits.shape, (N, num_nodes, self.max_arity))

        arg_logits2 = torch.transpose(arg_logits, 1, 2)
        assertEqual(arg_logits2.shape, (N, self.max_arity, num_nodes))

        if greedy:
            arg_idxs = torch.argmax(arg_logits2, dim=2)
        else:
            arg_idxs = Categorical(logits=arg_logits2).sample()
        assertEqual(arg_idxs.shape, (N, self.max_arity))

        return (arg_idxs, arg_logits2)


class AutoRegressiveChoiceNet(nn.Module):
    """
    Chooses args one by one, conditioning each arg choice on the args chosen so
    far.
    """
    def __init__(self, ops: Sequence[Op], node_dim: int, state_dim: int):
        super().__init__()
        self.ops = ops
        self.node_dim = node_dim
        self.state_dim = state_dim
        self.num_ops = len(ops)
        self.max_arity = max(op.arity for op in ops)

        # choosing args for op
        self.pointer_net = PointerNet2(
            input_dim=self.node_dim,
            # concat [state, op_one_hot, args_so_far_embeddings]
            query_dim=self.state_dim + self.num_ops + self.max_arity +
            self.max_arity * self.node_dim,
            hidden_dim=256,
            num_hidden=1,
        )
        # for choosing arguments when we haven't chosen anything yet
        self.blank_embed = torch.zeros(node_dim)

    def forward(
        self,
        op_idx: int,
        state_embed: Tensor,
        node_embed_list: List[Tensor],
        greedy: bool = False,
    ) -> Tuple[Tuple[int, ...], Tensor]:
        """
        Use attention over ValueNodes to choose arguments for op (may be None).
        This is implemented with the pointer network.

        Returns a tuple (args, args_logits_tensor).
            - args is a tuple of ValueNodes chosen (or None)
            - args_logits_tensor is a Tensor of shape (self.max_arity, N)
              showing the probability of choosing each node in the input list,
              used for doing loss updates.

        """

        args_logits = []
        arg_idxs = []
        args_embed: List[Tensor] = [self.blank_embed] * self.max_arity
        one_hot_op = F.one_hot(torch.tensor(op_idx), num_classes=self.num_ops)
        N = len(node_embed_list)
        nodes_embed = torch.stack(node_embed_list)

        assertEqual(nodes_embed.shape, (N, self.node_dim))
        assertEqual(one_hot_op.shape, (self.num_ops, ))
        assertEqual(state_embed.shape, (self.state_dim, ))
        for n in node_embed_list:
            assertEqual(n.shape, (self.node_dim, ))

        # chose args one at a time via pointer net
        for i in range(self.ops[op_idx].arity):
            idx_one_hot = F.one_hot(torch.tensor(i),
                                    num_classes=self.max_arity)
            args_tensor = torch.cat(args_embed)
            assertEqual(args_tensor.shape, (self.max_arity * self.node_dim, ))

            queries = torch.cat(
                [state_embed, one_hot_op, idx_one_hot, args_tensor])
            assertEqual(queries.shape,
                        (self.state_dim + self.num_ops + self.max_arity +
                         self.node_dim * self.max_arity, ))

            arg_logits = self.pointer_net(inputs=nodes_embed, queries=queries)
            assertEqual(arg_logits.shape, (N, ))

            args_logits.append(arg_logits)

            if greedy:
                arg_idx = torch.argmax(arg_logits).item()
            else:
                arg_idx = Categorical(logits=arg_logits).sample().item()

            assert isinstance(arg_idx, int)

            arg_idxs.append(arg_idx)

            args_embed[i] = node_embed_list[arg_idx]

        args_logits_tens = torch.stack(args_logits)
        # this shape expected by loss function
        assertEqual(args_logits_tens.shape, (self.max_arity, N))
        return tuple(arg_idxs), args_logits_tens


class BatchedDeepSetNet(nn.Module):
    def __init__(self,
                 element_dim: int,
                 set_dim: int,
                 hidden_dim: int,
                 presum_num_layers=1,
                 postsum_num_layers=1):
        super().__init__()

        self.element_dim = element_dim
        self.set_dim = set_dim
        self.hidden_dim = hidden_dim

        self.presum_net = FC(
            input_dim=element_dim,
            output_dim=hidden_dim,
            hidden_dim=hidden_dim,
            # output is one of the hidden for overall
            # architecture, so do one less
            num_hidden=presum_num_layers - 1)
        self.postsum_net = FC(input_dim=hidden_dim,
                              output_dim=set_dim,
                              hidden_dim=hidden_dim,
                              num_hidden=postsum_num_layers - 1)
        # self.finalize()

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters()
                       if p.requires_grad)

    def forward(self, batch_psg_embeddings: Tensor):
        # node_embeddings = self.tensor(node_embeddings)

        N = batch_psg_embeddings.shape[0]
        assertEqual(batch_psg_embeddings.shape[2], self.element_dim)

        presum = F.relu(self.presum_net(batch_psg_embeddings))
        assertEqual(presum.shape, (N, presum.shape[1], self.hidden_dim))

        postsum = torch.sum(presum, dim=1)
        assertEqual(postsum.shape, (N, self.hidden_dim))

        out = self.postsum_net(postsum)
        assertEqual(out.shape, (N, self.set_dim))
        return out


class PolicyNet(nn.Module):
    def __init__(self,
                 ops: Sequence[Op],
                 node_dim,
                 state_dim,
                 arg_choice_net: nn.Module,
                 node_embed_net: nn.Module,
                 use_cuda: bool = True,
                 max_nodes: int = 0):
        super().__init__()
        self.ops = ops
        self.num_ops = len(ops)
        self.node_dim = node_dim
        self.state_dim = state_dim
        self.node_embed_net = node_embed_net
        self.use_cuda = use_cuda
        self.max_nodes = max_nodes

        # for embedding the state
        self.deepset_net = BatchedDeepSetNet(element_dim=self.node_dim,
                                             hidden_dim=self.state_dim,
                                             presum_num_layers=1,
                                             postsum_num_layers=1,
                                             set_dim=self.state_dim)
        # for choosing op.
        self.op_choice_linear = nn.Linear(self.state_dim, self.num_ops)
        self.arg_choice_net: nn.Module = arg_choice_net
        assert arg_choice_net.node_dim == self.node_dim, (
            'subnets all need to coordinate using the same node_dim')
        assert arg_choice_net.ops == self.ops, (
            'subnets all neet to coordinate using the same ops')

        self.op_choice2 = nn.Linear(3 * 102, 3)
        self.arg_choice2 = nn.Linear(3 * 102, 6)

        if self.use_cuda:
            self.cuda()

    def forward(self,
                batch: List[ProgramSearchGraph],
                greedy: bool = False) -> PolicyPred:
        N = len(batch)

        if not self.max_nodes:
            max_nodes = max(len(psg.get_value_nodes()) for psg in batch)
        else:
            max_nodes = self.max_nodes

        batch_psg_embeddings = torch.zeros(N, max_nodes, self.node_embed_net.dim)
        if self.use_cuda:
            batch_psg_embeddings = batch_psg_embeddings.cuda()

        for j, psg in enumerate(batch):
            assert len(psg.get_value_nodes()) <= max_nodes
            for k, node in enumerate(psg.get_value_nodes()):
                batch_psg_embeddings[j, k, :] = self.node_embed_net(node, psg.is_grounded(node))

        # not sure if need it a second time
        if self.use_cuda:
            batch_psg_embeddings = batch_psg_embeddings.cuda()

        # print(f"batch_psg_embeddings: {batch_psg_embeddings.shape}")
        assertEqual(batch_psg_embeddings.shape,
                    (N, batch_psg_embeddings.shape[1], self.node_dim))

        assertEqual(batch_psg_embeddings.shape,
                    (N, 3, 102))

        inp = F.relu(torch.flatten(batch_psg_embeddings, start_dim=1))
        assertEqual(inp.shape, (N, 3 * 102))

        op_logits = self.op_choice2(inp)
        assertEqual(op_logits.shape, (N, self.num_ops))

        if greedy:
            op_idxs = torch.argmax(op_logits, dim=1)
        else:
            op_idxs = Categorical(logits=op_logits).sample()
        # assert isinstance(op_idxs, int)  # for type-checking

        arg_logits = self.arg_choice2(inp)
        assertEqual(arg_logits.shape, (N, 6))
        arg_logits2 = arg_logits.view(N, 2, 3)
        assertEqual(arg_logits2.shape, (N, self.arg_choice_net.max_arity, 3))

        if greedy:
            arg_idxs = torch.argmax(arg_logits2, dim=2)
        else:
            arg_idxs = Categorical(logits=arg_logits2).sample()
        assertEqual(arg_idxs.shape, (N, self.arg_choice_net.max_arity))

        return PolicyPred(op_idxs, arg_idxs, op_logits, arg_logits2)  # , outs

    def forward_old(self,
                    batch: List[ProgramSearchGraph],
                    greedy: bool = False) -> PolicyPred:

        N = len(batch)

        if not self.max_nodes:
            max_nodes = max(len(psg.get_value_nodes()) for psg in batch)
        else:
            max_nodes = self.max_nodes

        batch_psg_embeddings = torch.zeros(N, max_nodes, self.node_embed_net.dim)
        if self.use_cuda:
            batch_psg_embeddings = batch_psg_embeddings.cuda()

        for j, psg in enumerate(batch):
            assert len(psg.get_value_nodes()) <= max_nodes
            for k, node in enumerate(psg.get_value_nodes()):
                batch_psg_embeddings[j, k, :] = self.node_embed_net(node, psg.is_grounded(node))

        # not sure if need it a second time
        if self.use_cuda:
            batch_psg_embeddings = batch_psg_embeddings.cuda()

        # print(f"batch_psg_embeddings: {batch_psg_embeddings}")
        assertEqual(batch_psg_embeddings.shape,
                    (N, batch_psg_embeddings.shape[1], self.node_dim))
        state_embeds: Tensor = self.deepset_net(batch_psg_embeddings)
        assertEqual(state_embeds.shape, (N, self.state_dim))

        op_logits = self.op_choice_linear(F.relu(state_embeds))
        assertEqual(op_logits.shape, (N, self.num_ops))

        if greedy:
            op_idxs = torch.argmax(op_logits, dim=1)
        else:
            op_idxs = Categorical(logits=op_logits).sample()
        # assert isinstance(op_idxs, int)  # for type-checking

        # next step: choose the arguments
        arg_idxs, arg_logits = self.arg_choice_net(op_idxs=op_idxs,
                                                   state_embeds=state_embeds,
                                                   psg_embeddings=batch_psg_embeddings,
                                                   greedy=greedy)
        # nodes = state.get_value_nodes()
        # args = [nodes[idx] for idx in arg_idxs]

        # outs = []
        # for i in range(N):
        #     action = SynthEnvAction(op_idx[i].item(), tuple(arg_idxs[i].tolist()))
        #     outs.append(PolicyPred(action, op_logits[i], arg_logits[i]))

        return PolicyPred(op_idxs, arg_idxs, op_logits, arg_logits)  # , outs


class PolicyNet24(nn.Module):
    """
    Simplified version for last-minute trying to get 24 stuff to work.
    """
    def __init__(self,
                 ops: Sequence[Op],
                 node_dim,
                 node_embed_net,
                 max_arity: int = 2,
                 use_cuda: bool = True,
                 max_nodes: int = 0):
        super().__init__()
        self.ops = ops
        self.num_ops = len(ops)
        self.node_dim = node_dim
        self.use_cuda = use_cuda
        self.max_nodes = max_nodes
        self.max_arity = max_arity

        ArgChoiceNet = namedtuple('ArgChoiceNet', 'max_arity')
        self.arg_choice_net = ArgChoiceNet(max_arity=2)

        self.node_embed_net = node_embed_net

        self.inter_dim = 128
        self.fc_net = FC(input_dim=self.max_nodes * self.node_dim,
                         output_dim=self.inter_dim,
                         hidden_dim=64,
                         num_hidden=2)

        self.op_net = nn.Linear(self.inter_dim, self.num_ops)
        self.arg_net = nn.Linear(self.inter_dim + self.num_ops, self.max_arity * self.max_nodes)

        if self.use_cuda:
            self.cuda()

    def forward(self,
                batch: List[ProgramSearchGraph],
                greedy: bool = False) -> PolicyPred:
        N = len(batch)

        batch_psg_embeddings = torch.zeros(N, self.max_nodes, self.node_embed_net.dim)
        if self.use_cuda:
            batch_psg_embeddings = batch_psg_embeddings.cuda()

        for j, psg in enumerate(batch):
            assert len(psg.get_value_nodes()) <= self.max_nodes
            for k, node in enumerate(psg.get_value_nodes()):
                batch_psg_embeddings[j, k, :] = self.node_embed_net(node, psg.is_grounded(node))

        # not sure if need it a second time
        if self.use_cuda:
            batch_psg_embeddings = batch_psg_embeddings.cuda()

        # print(f"batch_psg_embeddings: {batch_psg_embeddings.shape}")
        assertEqual(batch_psg_embeddings.shape,
                    (N, batch_psg_embeddings.shape[1], self.node_dim))

        fc_input = torch.flatten(batch_psg_embeddings, start_dim=1)
        assertEqual(fc_input.shape, (N, self.node_dim * self.max_nodes))
        inter_result = F.relu(self.fc_net(fc_input))
        assertEqual(inter_result.shape, (N, self.inter_dim))

        op_logits = self.op_net(inter_result)
        assertEqual(op_logits.shape, (N, self.num_ops))

        if greedy:
            op_idxs = torch.argmax(op_logits, dim=1)
        else:
            op_idxs = Categorical(logits=op_logits).sample()
        assertEqual(op_idxs.shape, (N, ))

        op_one_hot = F.one_hot(op_idxs, num_classes=self.num_ops)
        assertEqual(op_one_hot.shape, (N, self.num_ops))

        in_tensor = torch.cat([inter_result, op_one_hot], dim=1)
        assertEqual(in_tensor.shape,
                    (N, self.inter_dim + self.num_ops))

        arg_logits = self.arg_net(in_tensor)
        assertEqual(arg_logits.shape, (N, self.max_arity * self.max_nodes))
        arg_logits = arg_logits.view(N, self.max_arity, self.max_nodes)

        assertEqual(arg_logits.shape, (N, self.max_arity, self.max_nodes))

        if greedy:
            arg_idxs = torch.argmax(arg_logits, dim=2)
        else:
            arg_idxs = Categorical(logits=arg_logits).sample()
        assertEqual(arg_idxs.shape, (N, self.max_arity))

        return PolicyPred(op_idxs, arg_idxs, op_logits, arg_logits)


def policy_net_24(ops: Sequence[Op],
                  max_int: int,
                  state_dim: int = 128,
                  use_cuda: bool = False,
                  max_nodes: int = 0) -> PolicyNet:
    node_embed_net = TwentyFourNodeEmbedNet(max_int)
    node_dim = node_embed_net.dim
    arg_choice_cls = DirectChoiceNet
    # arg_choice_cls = AutoRegressiveChoiceNet

    arg_choice_net = arg_choice_cls(ops=ops,
                                    node_dim=node_dim,
                                    state_dim=state_dim)
    policy_net = PolicyNet(ops=ops,
                           node_dim=node_dim,
                           state_dim=state_dim,
                           arg_choice_net=arg_choice_net,
                           node_embed_net=node_embed_net,
                           use_cuda=use_cuda,
                           max_nodes=max_nodes)
    return policy_net


def policy_net_arc(ops: Sequence[Op],
                   state_dim: int = 256,
                   use_cuda: bool = False,
                   max_nodes: int = 0) -> PolicyNet:
    node_embed_net = ArcNodeEmbedNetGridsOnly(dim=state_dim, use_cuda=use_cuda)
    node_dim = node_embed_net.dim
    arg_choice_cls = DirectChoiceNet
    # arg_choice_cls = AutoRegressiveChoiceNet

    arg_choice_net = arg_choice_cls(ops=ops,
                                    node_dim=node_dim,
                                    state_dim=state_dim)
    policy_net = PolicyNet(ops=ops,
                           node_dim=node_dim,
                           state_dim=state_dim,
                           arg_choice_net=arg_choice_net,
                           node_embed_net=node_embed_net,
                           use_cuda=use_cuda,
                           max_nodes=max_nodes)
    return policy_net


def policy_net_24_alt(ops: Sequence[Op],
                      max_int: int,
                      state_dim: int = 128,  # not used
                      use_cuda: bool = False,
                      max_nodes: int = 0) -> PolicyNet:
    node_embed_net = TwentyFourNodeEmbedNet(max_int)
    node_dim = node_embed_net.dim
    policy_net = PolicyNet24(ops=ops,
                             node_dim=node_dim,
                             node_embed_net=node_embed_net,
                             use_cuda=use_cuda,
                             max_nodes=max_nodes)
    return policy_net
