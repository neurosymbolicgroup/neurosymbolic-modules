from bidir.primitives.types import Grid
from rl.operations import Op
from rl.program_search_graph import ValueNode, ProgramSearchGraph
from typing import List, Tuple, Optional, Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# SynthAction = Tuple[Op, Tuple[Optional[ValueNode], ...]]
from rl.environment import SynthAction


class PolicyNet(nn.Module):
    def __init__(self, ops: List[Op]):
        self.ops = ops
        self.op_dict = dict(zip(ops, range(len(ops))))
        self.O = len(ops)
        # dimensionality of the valuenode embeddings
        self.D = 256
        # dimensionality of the state embedding
        self.S = 512
        # for embedding the state DeepSet-style
        self.nodeset_linear = nn.Linear(self.D, self.S)
        # for choosing op
        self.op_choice_linear = nn.Linear(self.S, self.O)
        # used in pointer net. TODO: put in another class?
        self.W1 = nn.Linear(self.D, self.D)
        # takes in concat of (state_embed, op_one_one, and arg_list_embed)
        self.W2 = nn.Linear(self.S + self.O + self.D, self.D)
        self.V = nn.Linear(self.D, 1)
        self.none_embed = torch.zeros(self.D)

    def one_hot_op(self, op: Op):
        ix = self.op_dict[op]
        return F.one_hot(torch.tensor(ix))

    def forward(self, state: ProgramSearchGraph) -> SynthAction:
        # TODO: is this valid?
        return self.choose_action(state)

    def choose_action(self, state) -> SynthAction:
        nodes: List[ValueNode] = state.get_value_nodes()

        embedded_nodes: List[Tensor] = [self.embed(node) for node in nodes]
        embedded_state: Tensor = self.embed_node_set(embedded_nodes)
        assert embedded_state.shape == (self.S, )
        # TODO: nonlinearity?
        op_logits = self.op_choice_linear(embedded_state)
        assert op_logits.shape == (self.O, )
        op_ix = torch.argmax(op_logits).item()
        assert isinstance(op_ix, int)  # for type-checking
        op_chosen = self.ops[op_ix]
        # next step: choose the arguments
        args = self.choose_args(op=op_chosen,
                                state_embed=embedded_state,
                                node_embed_list=embedded_nodes,
                                nodes=nodes)
        return op_chosen, args

    def choose_args(self, op: Op, state_embed: Tensor,
                    node_embed_list: List[Tensor],
                    nodes: List[ValueNode]) -> Tuple[Optional[ValueNode], ...]:
        """
        Use attention over value nodes to choose one for each argument.
        Condition on arguments already chosen?
        We will use a pointer net.
        See https://arxiv.org/pdf/1506.03134.pdf section 2.3.
        e_i is the node embedding. d_i will be a aoncatenation of the state
        embedding, the op one-hot encoding, and an embedding a list of the args
        chosen so far.
        """
        args: List[Optional[ValueNode]] = []
        one_hot_op = self.one_hot_op(op)
        N = len(node_embed_list)
        # add None as an arg option
        # done with '+' to prevent mutating original list
        node_embed_list = node_embed_list + [self.none_embed]
        # wrap RHS with list() for type-checking sake, see strategy 2 here:
        # https://mypy.readthedocs.io/en/latest/common_issues.html#variance
        nodes_with_none: List[Optional[ValueNode]] = list(nodes)
        nodes_with_none.append(None)
        nodes_embed = torch.stack(node_embed_list)
        assert nodes_embed.shape == (N, self.D)

        assert one_hot_op.shape == self.O
        assert state_embed.shape == self.S
        assert all(n.shape == self.D for n in node_embed_list)

        for i in range(op.arity):
            # TODO: could make this more efficient, instead of recalculating
            # each time we append
            args_embed = self.embed_by_type(args)
            assert args_embed.shape == (self.D, )
            d = torch.cat([state_embed, one_hot_op, args_embed])
            assert d.shape == (self.S + self.O + self.D)
            w1_out = self.W1(nodes_embed)
            assert w1_out.shape == (N, self.D)
            w2_out = self.W2(d)
            assert w2_out.shape == (self.D, )
            w2_repeated = w2_out.repeat(N, 1)
            assert w2_repeated.shape == (N, self.D)
            w1_plus_w2 = w2_out + w2_repeated
            u_i = self.V(nn.Tanh(w1_plus_w2))
            assert u_i.shape == (N, )
            # TODO sample using temperature instead?
            arg_ix = torch.argmax(u_i).item()
            assert isinstance(arg_ix, int)  # for type checking
            node_choice = nodes_with_none[arg_ix]
            args.append(node_choice)

        return tuple(args)

    def embed(self, node: ValueNode) -> Tensor:
        # embeds each node via type-based
        examples: Tuple = node._value
        # embedding example list same way we would embed any other list
        return self.embed_by_type(examples)

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
            # I forget how we're handling ints/bools/colors.
            # how will we handle colors?
            raise NotImplementedError

    def embed_tensor_list(self, emb_list) -> Tensor:
        # TODO: run through an LSTM or something?
        raise NotImplementedError

    def embed_grid(self, grid: Grid) -> Tensor:
        raise NotImplementedError
        # TODO: plug in simple CNN

    def embed_node_set(self, node_embeddings: List[Tensor]) -> Tensor:
        # yup, it's as simple as that.
        summed = sum(node_embeddings)
        # TODO: nonlinearity needed?
        return self.nodeset_linear(summed)
