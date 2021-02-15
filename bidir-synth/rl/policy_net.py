from typing import List, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions.categorical import Categorical

from bidir.primitives.types import Grid
from bidir.utils import assertEqual

from modules.synth_modules import PointerNet, DeepSetNet
from modules.base_modules import FC
from rl.ops.operations import Op
from rl.program_search_graph import ValueNode, ProgramSearchGraph


class PolicyPred(NamedTuple):
    op_idx: int
    arg_idxs: Tuple[int, ...]
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


class ArgChoiceNet(nn.Module):
    def __init__(self, ops: List[Op], node_dim: int, state_dim: int):
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
    ) -> Tuple[Tuple[int, ...], Tensor]:
        raise NotImplementedError


class DirectChoiceNet(ArgChoiceNet):
    def __init__(self, ops: List[Op], node_dim: int, state_dim: int):
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
    ) -> Tuple[Tuple[int, ...], Tensor]:
        """
        Equivalent to a pointer net, but chooses args all at once.
        Much easier to understand, too.
        """

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
        query = query.repeat(N, 1)
        in_tensor = torch.cat([query, nodes_embed], dim=1)
        assertEqual(in_tensor.shape,
                    (N, self.state_dim + self.num_ops + self.node_dim))

        arg_logits = self.args_net(in_tensor)
        assertEqual(arg_logits.shape, (N, self.max_arity))

        arg_logits2 = torch.transpose(arg_logits, 0, 1)
        arg_idxs = Categorical(logits=arg_logits2).sample()
        assertEqual(arg_idxs.shape, (self.max_arity, ))

        return (tuple(arg_idxs.tolist()), arg_logits2)


class AutoRegressiveChoiceNet(ArgChoiceNet):
    """
    Chooses args one by one, conditioning each arg choice on the args chosen so
    far.
    """
    def __init__(self, ops: List[Op], node_dim: int, state_dim: int):
        super().__init__(ops, node_dim, state_dim)
        # choosing args for op
        self.pointer_net = PointerNet(
            input_dim=self.node_dim,
            # concat [state, op_one_hot, args_so_far_embeddings]
            query_dim=self.state_dim + self.num_ops + self.max_arity +
            self.max_arity * self.node_dim,
            hidden_dim=64,
        )
        # for choosing arguments when we haven't chosen anything yet
        self.blank_embed = torch.zeros(node_dim)

    def forward(
        self,
        op_idx: int,
        state_embed: Tensor,
        node_embed_list: List[Tensor],
    ) -> Tuple[Tuple[int, ...], Tensor]:
        """
        Use attention over ValueNodes to choose arguments for op (may be None).
        This is implemented with the pointer network.

        Returns a tuple (args, args_logits_tensor).
            - args is a tuple of ValueNodes chosen (or None)
            - args_logits_tensor is a Tensor of shape (self.max_arity, N)
              showing the probability of choosing each node in the input list,
              used for doing loss updates.

        See https://arxiv.org/pdf/1506.03134.pdf section 2.3.

        The pointer network uses additive attention. Since the paper came out,
        dot-product attention has become preferred, so maybe we should do that.

        Using the paper terminology, the inputs are:
        e_j is the node embedding.
        d_i is a aoncatenation of:
            - the state embedding
            - the op one-hot encoding
            - the arg index one-hot
            - the list of args chosen so far
        """
        # TODO: need to test this out
        op = self.ops[op_idx]
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
        for i in range(op.arity):
            idx_one_hot = F.one_hot(torch.tensor(i),
                                    num_classes=self.max_arity)
            # recompute each time as we add args chosen
            # TODO: make more efficient, perhaps
            args_tensor = torch.cat(args_embed)
            assertEqual(args_tensor.shape, (self.max_arity * self.node_dim, ))

            queries = torch.cat(
                [state_embed, one_hot_op, idx_one_hot, args_tensor])
            # print(f"queries: {queries}")

            assertEqual(queries.shape,
                        (self.state_dim + self.num_ops + self.max_arity +
                         self.node_dim * self.max_arity, ))
            arg_logits = self.pointer_net(inputs=nodes_embed, queries=queries)
            # print(f"arg_logits: {arg_logits}")
            args_logits.append(arg_logits)
            assertEqual(arg_logits.shape, (N, ))
            # arg_idx = torch.argmax(arg_logits).item()
            arg_idx = Categorical(logits=arg_logits).sample()
            arg_idxs.append(arg_idx)
            assert isinstance(arg_idx, int)  # for type checking
            args_embed[i] = node_embed_list[arg_idx]

        args_logits_tens = torch.stack(args_logits)
        # this shape expected by loss function
        assertEqual(args_logits_tens.shape, (self.max_arity, N))
        return tuple(arg_idxs), args_logits_tens


class PolicyNet(nn.Module):
    def __init__(self, ops: List[Op], node_dim, state_dim,
                 node_embed_net: NodeEmbedNet, arg_choice_net: ArgChoiceNet):
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
                                      set_dim=self.state_dim)
        # for choosing op.
        self.op_choice_linear = nn.Linear(self.state_dim, self.num_ops)

        assert node_embed_net.dim == self.node_dim, (
            'subnets all need ', 'to coordinate using the same node_dim')
        self.arg_choice_net = arg_choice_net
        assert arg_choice_net.node_dim == self.node_dim, (
            'subnets all need ', 'to coordinate using the same node_dim')
        assert arg_choice_net.ops == self.ops, (
            'subnets all neet ', 'to coordinate using the same ops')

    def forward(self, state: ProgramSearchGraph) -> PolicyPred:

        nodes: List[ValueNode] = state.get_value_nodes()
        embedded_nodes: List[Tensor] = [
            self.node_embed_net(node, state.is_grounded(node))
            for node in nodes
        ]

        node_embed_tens = torch.stack(embedded_nodes)
        state_embed: Tensor = self.deepset_net(node_embed_tens)

        assertEqual(state_embed.shape, (self.state_dim, ))

        # TODO: place nonlinearities with more intentionality
        op_logits = self.op_choice_linear(F.relu(state_embed))

        assertEqual(op_logits.shape, (self.num_ops, ))

        # TODO: sample stochastically instead
        # op_idx = torch.argmax(op_logits).item()
        op_idx = Categorical(logits=op_logits).sample().item()
        assert isinstance(op_idx, int)  # for type-checking

        # next step: choose the arguments
        arg_idxs, arg_logits = self.arg_choice_net(
            op_idx=op_idx,
            state_embed=state_embed,
            node_embed_list=embedded_nodes)

        return PolicyPred(op_idx, arg_idxs, op_logits, arg_logits)


def policy_net_24(ops: List[Op],
                  max_int: int,
                  state_dim: int = 512) -> PolicyNet:
    node_embed_net = TwentyFourNodeEmbedNet(max_int)
    node_dim = node_embed_net.dim
    arg_choice_net = DirectChoiceNet(ops, node_dim, state_dim)
    policy_net = PolicyNet(ops, node_dim, state_dim, node_embed_net,
                           arg_choice_net)
    return policy_net
