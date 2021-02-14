from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from bidir.primitives.types import Grid
from bidir.utils import assertEqual

from modules.synth_modules import PointerNet, DeepSetNet
from modules.base_modules import FC
from rl.environment import SynthEnvAction
from rl.ops.operations import Op
from rl.program_search_graph import ValueNode, ProgramSearchGraph

# SynthAction = Tuple[Op, Tuple[Optional[ValueNode], ...]]


# TODO: factor out into base policynet
class PolicyNet(nn.Module):
    def __init__(self, ops: List[Op], node_dim=256, state_dim=512):
        super().__init__()
        self.ops = ops
        self.op_to_idx = {op: idx for (idx, op) in enumerate(ops)}
        self.max_arity = max(op.arity for op in ops)
        self.O = len(ops)

        # dimensionality of the valuenode embeddings
        # names: type_embed_dim, node_aux_dim
        self.type_embed_dim = node_dim
        self.node_aux_dim = 1  # extra dim to encoded groundedness
        self.D = self.type_embed_dim + self.node_aux_dim

        # dimensionality of the state embedding
        self.S = state_dim
        # for choosing arguments when we haven't chosen anything yet
        self.blank_embed = torch.zeros(self.D)
        # for embedding the state
        self.deepset_net = DeepSetNet(element_dim=self.D,
                                      hidden_dim=self.S,
                                      set_dim=self.S)

        # for choosing op
        self.op_choice_linear = nn.Linear(self.S, self.O)
        # choosing args for op
        self.pointer_net = PointerNet(
            input_dim=self.D,
            # concat [state, op_one_hot, args_so_far_embeddings]
            query_dim=self.S + self.O + self.max_arity +
            self.max_arity * self.D,
            hidden_dim=64,
        )

        # TODO: will have to turn grid numpy array into torch tensor with
        # different channel for each color
        # self.CNN = CNN(in_channels=len(Color), output_dim=self.D)
        # takes in sequence of embeddings, and produces an embedding of same
        # dimensionality.
        # self.LSTM = LSTM(input_dim=self.D, hidden_dim=64, output_dim=self.D)

    def one_hot_op(self, op: Op):
        idx = self.op_to_idx[op]
        return F.one_hot(torch.tensor(idx), num_classes=self.O)

    def forward(
        self, state: ProgramSearchGraph
    ) -> Tuple[SynthEnvAction, Tuple[Tensor, Tensor]]:
        # TODO: is this valid?
        return self.choose_action(state)

    def choose_action(
        self, state: ProgramSearchGraph
    ) -> Tuple[SynthEnvAction, Tuple[Tensor, Tensor]]:
        nodes: List[ValueNode] = state.get_value_nodes()

        embedded_nodes: List[Tensor] = [
            self.embed(node, state.is_grounded(node)) for node in nodes
        ]

        node_embed_tens = torch.stack(embedded_nodes)
        state_embed: Tensor = self.deepset_net(node_embed_tens)
        assertEqual(state_embed.shape, (self.S, ))
        # TODO: place nonlinearities with more intentionality
        op_logits = self.op_choice_linear(F.relu(state_embed))
        assertEqual(op_logits.shape, (self.O, ))
        # TODO: sample stochastically instead
        op_idx = torch.argmax(op_logits).item()
        assert isinstance(op_idx, int)  # for type-checking
        op_chosen = self.ops[op_idx]
        # next step: choose the arguments
        args, args_logits = self.choose_args(op=op_chosen,
                                             state_embed=state_embed,
                                             node_embed_list=embedded_nodes,
                                             nodes=nodes)
        return SynthEnvAction(op_idx, args), (op_logits, args_logits)

    def choose_args(
        self,
        op: Op,
        state_embed: Tensor,
        node_embed_list: List[Tensor],
        nodes: List[ValueNode],
    ) -> Tuple[Tuple[ValueNode, ...], Tensor]:
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
        args_logits = []
        args: List[ValueNode] = []
        args_embed: List[Tensor] = [self.blank_embed] * self.max_arity
        one_hot_op = self.one_hot_op(op)
        N = len(node_embed_list)
        nodes_embed = torch.stack(node_embed_list)

        assertEqual(nodes_embed.shape, (N, self.D))
        assertEqual(one_hot_op.shape, (self.O, ))
        assertEqual(state_embed.shape, (self.S, ))
        for n in node_embed_list:
            assertEqual(n.shape, (self.D, ))

        # chose args one at a time via pointer net
        for i in range(op.arity):
            idx_one_hot = F.one_hot(torch.tensor(i),
                                    num_classes=self.max_arity)
            # recompute each time as we add args chosen
            # TODO: make more efficient, perhaps
            args_tensor = torch.cat(args_embed)
            assertEqual(args_tensor.shape, (self.max_arity * self.D, ))

            queries = torch.cat(
                [state_embed, one_hot_op, idx_one_hot, args_tensor])
            # print(f"queries: {queries}")

            assertEqual(
                queries.shape,
                (self.S + self.O + self.max_arity + self.D * self.max_arity, ))
            arg_logits = self.pointer_net(inputs=nodes_embed, queries=queries)
            # print(f"arg_logits: {arg_logits}")
            args_logits.append(arg_logits)
            assertEqual(arg_logits.shape, (N, ))
            # TODO sample stochastically instead
            arg_idx = torch.argmax(arg_logits).item()
            assert isinstance(arg_idx, int)  # for type checking
            node_choice = nodes[arg_idx]
            args.append(node_choice)
            args_embed[i] = node_embed_list[arg_idx]

        args_logits_tensor = torch.transpose(torch.stack(args_logits), 0, 1)
        # this shape expected by loss function
        assertEqual(args_logits_tensor.shape, (N, self.max_arity))
        return tuple(args), args_logits_tensor

    def embed(self, node: ValueNode, is_grounded: bool) -> Tensor:
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

        out = torch.cat(
            [self.embed_by_type(examples), grounded_tensor])
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


class PolicyNet24(PolicyNet):
    def __init__(self, ops: List[Op], node_dim, state_dim):

        super().__init__(ops, node_dim=node_dim, state_dim=state_dim)
        self.args_net = FC(input_dim=self.S + self.O + self.D,
                           output_dim=self.max_arity,
                           num_hidden=1,
                           hidden_dim=256)

    def choose_action2(
        self, state: ProgramSearchGraph
    ) -> Tuple[SynthEnvAction, Tuple[Tensor, Tensor]]:
        node_embeds = [
            self.embed(node, state.is_grounded(node))
            for i, node in enumerate(state.get_value_nodes())
        ]
        node_embed_tens = torch.stack(node_embeds)
        state_embed = self.deepset_net(node_embed_tens)
        op_logits = self.op_choice_linear(F.relu(state_embed))
        assertEqual(op_logits.shape, (self.O, ))
        # TODO: sample stochastically instead
        op_idx = torch.argmax(op_logits).item()
        assert isinstance(op_idx, int)  # for type-checking
        op_choice = self.ops[op_idx]

        (arg_choices, arg_logits) = self.choose_args2(op_choice, state_embed,
                                                      node_embeds,
                                                      state.get_value_nodes())

        return SynthEnvAction(op_idx, arg_choices), (op_logits, arg_logits)

    def choose_args2(
        self,
        op: Op,
        state_embed: Tensor,
        node_embed_list: List[Tensor],
        nodes: List[ValueNode],
    ) -> Tuple[Tuple[ValueNode, ...], Tensor]:
        op_one_hot = self.one_hot_op(op)
        N = len(node_embed_list)
        nodes_embed = torch.stack(node_embed_list)

        assertEqual(nodes_embed.shape, (N, self.D))
        assertEqual(op_one_hot.shape, (self.O, ))
        assertEqual(state_embed.shape, (self.S, ))
        for n in node_embed_list:
            assertEqual(n.shape, (self.D, ))

        # in tensor: (N, S + O + D)
        query = torch.cat([op_one_hot, state_embed])
        query = query.repeat(N, 1)
        in_tensor = torch.cat([query, nodes_embed], dim=1)
        assertEqual(in_tensor.shape, (N, self.S + self.O + self.D))
        arg_logits = self.args_net(in_tensor)
        assertEqual(arg_logits.shape, (N, self.max_arity))
        # TODO sample stochastically instead
        arg_idxs = torch.argmax(arg_logits, dim=0)
        arg_choices = tuple(nodes[idx] for idx in arg_idxs)
        return (arg_choices, arg_logits)

    def forward(self, state):
        return self.choose_action2(state)

    def embed_by_type(self, value) -> Tensor:
        assert isinstance(value, tuple)
        assertEqual(len(value), 1)
        n = value[0]
        assert isinstance(n, int)
        # values range from zero to MAX_INT inclusive
        out = F.one_hot(torch.tensor(n), num_classes=self.type_embed_dim)
        out = out.to(torch.float32)
        return out


class OpNet24(PolicyNet):
    def __init__(self, ops: List[Op], max_int: int):
        super().__init__(ops, node_dim=max_int + 1, state_dim=max_int + 1)

    def forward(self, state: ProgramSearchGraph) -> Tensor:
        node_embeds = torch.stack([
            self.embed(node, state.is_grounded(node))
            for node in state.get_value_nodes()
        ])
        state_embed = self.deepset_net(node_embeds)

        op_logits = self.op_choice_linear(F.relu(state_embed))
        assertEqual(op_logits.shape, (self.O, ))

        return op_logits

    def embed_by_type(self, value) -> Tensor:
        assert isinstance(value, tuple)
        assertEqual(len(value), 1)
        n = value[0]
        assert isinstance(n, int)

        # values range from zero to MAX_INT inclusive
        out = F.one_hot(torch.tensor(n), num_classes=self.type_embed_dim)
        out = out.to(torch.float32)
        return out
