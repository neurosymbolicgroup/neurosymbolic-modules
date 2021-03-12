from typing import List, Tuple, NamedTuple, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.categorical import Categorical

from bidir.primitives.types import Grid, MIN_COLOR, NUM_COLORS
from bidir.utils import assertEqual

from modules.synth_modules import DeepSetNet, PointerNet2
from modules.base_modules import FC
from rl.ops.operations import Op
from rl.environment import SynthEnvAction
from rl.program_search_graph import ValueNode, ProgramSearchGraph


def compute_out_size(in_size, model: nn.Module):
    """
    Compute output size of model for an input with size `in_size`.
    This is a utility function.
    """
    out = model(torch.Tensor(1, *in_size))  # type: ignore
    return int(np.prod(out.size()))


class PolicyPred(NamedTuple):
    action: SynthEnvAction
    op_logits: Tensor
    arg_logits: Tensor  # shape (max_arity, num_input_nodes)


class NodeEmbedNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, node: ValueNode, is_grounded: bool) -> Tensor:
        raise NotImplementedError


class TwentyFourNodeEmbedNet(NodeEmbedNet):
    def __init__(self, max_int):
        self.aux_dim = 1  # extra dim to encode groundedness
        self.embed_dim = max_int + 1
        super().__init__(dim=self.embed_dim + self.aux_dim)

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


class ArcNodeEmbedNet(nn.Module):
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


class ArcNodeEmbedNetGridsOnly(NodeEmbedNet):
    """Only embeds nodes with values of type Tuple[Grid]"""
    def __init__(self, dim, side_len=32):
        """
        dim is the output dimension
        side_len is what grids get padded/cropped to internally
        """
        super().__init__(dim=dim)
        self.aux_dim = 1  # extra dim to encode groundedness
        self.embed_dim = dim - self.aux_dim

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

        self.embedding_combiner = DeepSetNet(
            element_dim=self.CNN_output_size,
            hidden_dim=self.embed_dim,
            presum_num_layers=1,
            postsum_num_layers=1,
            set_dim=self.embed_dim,
        )

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

        out = torch.cat([
            combined_embeddings,
            torch.tensor([int(is_grounded)]),
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


class ArgChoiceNet(nn.Module):
    def __init__(self, ops: Sequence[Op], node_dim: int, state_dim: int):
        super().__init__()
        self.ops = ops
        self.node_dim = node_dim
        self.state_dim = state_dim
        self.num_ops = len(ops)
        self.max_arity = max(op.arity for op in ops)

    def forward(
        self,
        op_idx: int,
        state_embed: Tensor,
        node_embed_list: List[Tensor],
        greedy: bool = False,
    ) -> Tuple[Tuple[int, ...], Tensor]:
        raise NotImplementedError


class DirectChoiceNet(ArgChoiceNet):
    def __init__(self, ops: Sequence[Op], node_dim: int, state_dim: int):
        super().__init__(ops, node_dim, state_dim)
        self.args_net = FC(input_dim=self.state_dim + self.num_ops +
                           self.node_dim,
                           output_dim=self.max_arity,
                           num_hidden=1,
                           hidden_dim=256)

    def forward(
        self,
        op_idx: int,
        state_embed: Tensor,
        node_embed_list: List[Tensor],
        greedy: bool = False,
    ) -> Tuple[Tuple[int, ...], Tensor]:
        """
        Equivalent to a pointer net, but chooses args all at once.
        Much easier to understand, too.
        """
        op_arity = self.ops[op_idx].arity

        op_one_hot = F.one_hot(torch.tensor(op_idx), num_classes=self.num_ops)
        # op_one_hot = self.tensor(op_one_hot)
        N = len(node_embed_list)
        nodes_embed = torch.stack(node_embed_list)
        # nodes_embed = self.tensor(nodes_embed)
        # state_embed = self.tensor(state_embed)

        assertEqual(nodes_embed.shape, (N, self.node_dim))
        assertEqual(op_one_hot.shape, (self.num_ops, ))
        assertEqual(state_embed.shape, (self.state_dim, ))
        for n in node_embed_list:
            assertEqual(n.shape, (self.node_dim, ))

        # in tensor: (N, state_dim + op_dim + node_dim)
        query = torch.cat([op_one_hot, state_embed])
        query = query.repeat(N, 1)
        in_tensor = torch.cat([query, nodes_embed], dim=1)
        assertEqual(in_tensor.shape,
                    (N, self.state_dim + self.num_ops + self.node_dim))

        # process each node if separate elements in a batch
        arg_logits = self.args_net(in_tensor)
        assertEqual(arg_logits.shape, (N, self.max_arity))

        arg_logits2 = torch.transpose(arg_logits, 0, 1)
        assertEqual(arg_logits2.shape, (self.max_arity, N))

        if greedy:
            arg_idxs = torch.argmax(arg_logits2, dim=1)
        else:
            arg_idxs = Categorical(logits=arg_logits2).sample()
        assertEqual(arg_idxs.shape, (self.max_arity, ))

        # return (tuple(arg_idxs.tolist()[:op_arity]), arg_logits2[:op_arity])
        # need full set of arg_idxs when doing supervised training, for now.
        return (tuple(arg_idxs.tolist()), arg_logits2)


class ChoiceNet2(ArgChoiceNet):
    def __init__(self, ops: Sequence[Op], node_dim: int, state_dim: int):
        super().__init__(ops, node_dim, state_dim)
        self.pointer_net0 = PointerNet2(input_dim=self.node_dim,
                                        query_dim=self.state_dim +
                                        self.num_ops,
                                        hidden_dim=256,
                                        num_hidden=1)
        self.pointer_net1 = PointerNet2(input_dim=self.node_dim,
                                        query_dim=self.state_dim +
                                        self.num_ops + self.node_dim,
                                        hidden_dim=256,
                                        num_hidden=1)

    def make_inputs(self, op_idx: int, state_embed: Tensor,
                    node_embed_list: List[Tensor]) -> Tuple[Tensor, Tensor]:
        assert self.max_arity == 2

        op_one_hot = F.one_hot(torch.tensor(op_idx), num_classes=self.num_ops)
        N = len(node_embed_list)
        nodes_embed = torch.stack(node_embed_list)

        assertEqual(nodes_embed.shape, (N, self.node_dim))
        assertEqual(op_one_hot.shape, (self.num_ops, ))
        assertEqual(state_embed.shape, (self.state_dim, ))
        for n in node_embed_list:
            assertEqual(n.shape, (self.node_dim, ))

        # in tensor: (N, state_dim + op_dim + node_dim)
        query = torch.cat([op_one_hot, state_embed])
        return nodes_embed, query

    def forward(
        self,
        op_idx: int,
        state_embed: Tensor,
        node_embed_list: List[Tensor],
        greedy: bool = False,
    ) -> Tuple[Tuple[int, ...], Tensor]:
        assert self.max_arity == 2
        N = len(node_embed_list)

        inputs, query = self.make_inputs(op_idx, state_embed, node_embed_list)
        arg_logits0 = self.pointer_net0(inputs, query)
        assertEqual(arg_logits0.shape, (N, ))

        if greedy:
            arg0_idx = torch.argmax(arg_logits0).item()
        else:
            arg0_idx = Categorical(logits=arg_logits0).sample().item()
        assert isinstance(arg0_idx, int)  # for type-checking

        first_arg = node_embed_list[arg0_idx]
        query = torch.cat([query, first_arg])

        arg_logits1 = self.pointer_net1(inputs, query)
        assertEqual(arg_logits1.shape, (N, ))

        if greedy:
            arg1_idx = torch.argmax(arg_logits1).item()
        else:
            arg1_idx = Categorical(logits=arg_logits1).sample().item()
        assert isinstance(arg1_idx, int)  # for type-checking

        arg_logits = torch.stack([arg_logits0, arg_logits1])
        assertEqual(arg_logits.shape, (self.max_arity, N))

        return (arg0_idx,
                arg1_idx), arg_logits  # TODO: Dynamically adjust arity


class AutoRegressiveChoiceNet(ArgChoiceNet):
    """
    Chooses args one by one, conditioning each arg choice on the args chosen so
    far.
    """
    def __init__(self, ops: Sequence[Op], node_dim: int, state_dim: int):
        super().__init__(ops, node_dim, state_dim)
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

        assert not greedy, 'greedy isnt implemented/tested yet'

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

            arg_idx = Categorical(logits=arg_logits).sample().item()
            arg_idxs.append(arg_idx)

            args_embed[i] = node_embed_list[arg_idx]

        args_logits_tens = torch.stack(args_logits)
        # this shape expected by loss function
        assertEqual(args_logits_tens.shape, (self.max_arity, N))
        return tuple(arg_idxs), args_logits_tens


class PolicyNet(nn.Module):
    def __init__(self,
                 ops: Sequence[Op],
                 node_dim,
                 state_dim,
                 node_embed_net: NodeEmbedNet,
                 arg_choice_net: ArgChoiceNet,
                 greedy_op: bool = False):
        super().__init__()
        self.ops = ops
        self.num_ops = len(ops)
        self.node_dim = node_dim
        self.state_dim = state_dim
        self.node_embed_net = node_embed_net
        self.arg_choice_net = arg_choice_net

        # for embedding the state
        self.deepset_net = DeepSetNet(element_dim=self.node_dim,
                                      hidden_dim=self.state_dim,
                                      presum_num_layers=1,
                                      postsum_num_layers=1,
                                      set_dim=self.state_dim)
        # for choosing op.
        self.op_choice_linear = nn.Linear(self.state_dim, self.num_ops)

        assert node_embed_net.dim == self.node_dim, (
            'subnets all need to coordinate using the same node_dim')
        self.arg_choice_net = arg_choice_net
        assert arg_choice_net.node_dim == self.node_dim, (
            'subnets all need to coordinate using the same node_dim')
        assert arg_choice_net.ops == self.ops, (
            'subnets all neet to coordinate using the same ops')

    def forward(self,
                state: ProgramSearchGraph,
                greedy: bool = False) -> PolicyPred:
        nodes: List[ValueNode] = state.get_value_nodes()
        embedded_nodes: List[Tensor] = [
            self.node_embed_net(node, state.is_grounded(node))
            for node in nodes
        ]

        node_embed_tens = torch.stack(embedded_nodes)
        state_embed: Tensor = self.deepset_net(node_embed_tens)
        assertEqual(state_embed.shape, (self.state_dim, ))

        op_logits = self.op_choice_linear(F.relu(state_embed))
        assertEqual(op_logits.shape, (self.num_ops, ))

        if greedy:
            op_idx = torch.argmax(op_logits).item()
        else:
            op_idx = Categorical(logits=op_logits).sample().item()
        assert isinstance(op_idx, int)  # for type-checking

        op = self.ops[op_idx]

        # next step: choose the arguments
        arg_idxs, arg_logits = self.arg_choice_net(
            op_idx=op_idx,
            state_embed=state_embed,
            node_embed_list=embedded_nodes,
            greedy=greedy)
        nodes = state.get_value_nodes()
        args = [nodes[idx] for idx in arg_idxs]

        action = SynthEnvAction(op_idx, arg_idxs)

        return PolicyPred(action, op_logits, arg_logits)


class OldArgChoiceNet(ArgChoiceNet):
    def __init__(self, ops: Sequence[Op], node_dim: int, state_dim: int):
        super().__init__(ops, node_dim, state_dim)
        self.args_net = FC(input_dim=self.state_dim + self.num_ops +
                           self.node_dim,
                           output_dim=self.max_arity,
                           num_hidden=1,
                           hidden_dim=256)

    def forward(
        self,
        op_idx: int,
        state_embed: Tensor,
        node_embed_list: List[Tensor],
        greedy: bool = False,
    ) -> Tuple[Tuple[int, ...], Tensor]:
        op_one_hot = F.one_hot(torch.tensor(op_idx), num_classes=self.num_ops)
        N = len(node_embed_list)
        nodes_embed = torch.stack(node_embed_list)

        assertEqual(nodes_embed.shape, (N, self.node_dim))
        assertEqual(op_one_hot.shape, (self.num_ops, ))
        assertEqual(state_embed.shape, (self.state_dim, ))
        for n in node_embed_list:
            assertEqual(n.shape, (self.node_dim, ))

        # in tensor: (N, S + O + D)
        query = torch.cat([op_one_hot, state_embed])
        query = query.repeat(N, 1)
        in_tensor = torch.cat([query, nodes_embed], dim=1)
        assertEqual(in_tensor.shape,
                    (N, self.state_dim + self.num_ops + self.node_dim))
        arg_logits = self.args_net(in_tensor)
        assertEqual(arg_logits.shape, (N, self.max_arity))
        # TODO sample stochastically instead
        arg_idxs = torch.argmax(arg_logits, dim=0)

        arg_logits = torch.transpose(arg_logits, 0, 1)
        # this shape expected
        assertEqual(arg_logits.shape, (self.max_arity, N))
        return (tuple(arg_idxs.tolist()), arg_logits)


def policy_net_24(ops: Sequence[Op],
                  max_int: int,
                  state_dim: int = 128) -> PolicyNet:
    node_embed_net = TwentyFourNodeEmbedNet(max_int)
    node_dim = node_embed_net.dim
    arg_choice_cls = DirectChoiceNet

    arg_choice_net = arg_choice_cls(ops=ops,
                                    node_dim=node_dim,
                                    state_dim=state_dim)
    policy_net = PolicyNet(ops=ops,
                           node_dim=node_dim,
                           state_dim=state_dim,
                           node_embed_net=node_embed_net,
                           arg_choice_net=arg_choice_net)
    return policy_net


def policy_net_arc_v1(
    ops: Sequence[Op],
    state_dim: int = 256,
) -> PolicyNet:
    node_embed_net = ArcNodeEmbedNetGridsOnly(dim=state_dim)
    node_dim = node_embed_net.dim
    arg_choice_cls = DirectChoiceNet

    arg_choice_net = arg_choice_cls(ops=ops,
                                    node_dim=node_dim,
                                    state_dim=state_dim)
    policy_net = PolicyNet(ops=ops,
                           node_dim=node_dim,
                           state_dim=state_dim,
                           node_embed_net=node_embed_net,
                           arg_choice_net=arg_choice_net)
    return policy_net
