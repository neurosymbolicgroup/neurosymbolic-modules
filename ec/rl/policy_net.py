from bidir.primitives.types import Grid, Array, Color
from typing import NewType
import torch
import torch.nn as nn

# ArcAction = Tuple[Op, Tuple[Optional[ValueNode], ...]]
from rl.environment import ArcAction


Tensor = torch.Tensor

class Policy:
    def __init__(self, ops: List[Op]):
        self.ops = ops
        self.O = len(ops)
        # dimensionality of the valuenode embeddings
        self.D = 512
        # dimensionality of the state embedding
        self.S = 512
        self.nodeset_linear = nn.Linear(self.D, self.S)
        self.op_choice_linear = nn.Linear(self.S, self.O)

    def choose_action(self, state) -> ArcAction:
        nodes: List[ValueNode] = state.get_value_nodes()

        embedded_nodes: List[Tensor] = [self.embed(node) for node in nodes]
        embedded_state: Tensor = self.embed_node_set(embedded_nodes)
        op_logits = self.op_logits(embedded_state)
        assert op_logits.shape = (self.O, )
        op_ix = torch.argmax(op_logits)
        op_chosen = self.ops[op_ix]
        # next step: choose the arguments
        args = choose_args(op_chosen, embedded_state, embedded_nodes)
        return op_chosen, args

    def choose_args(op: Op, state_embedding: Tensor, nodes_embedding: Tensor, nodes: List[ValueNode]) -> ArcAction:
        # use attention over value nodes to choose one for each argument.
        # condition on arguments already chosen?
        raise NotImplementedError

    def embed(node: ValueNode) -> Tensor:
        # embeds each node via type-based
        Tuple examples = node._value
        # embedding example list same way we would embed any other list
        return embed_by_type(examples)

    def embed_by_type(value):
        # possible types: Tuples/Lists, Grids, Arrays, bool/int/color.
        if isinstance(value, tuple) or isinstance(value, list):
            subembeddings = [embed_by_type(x) for x in values]
            return embed_tensor_list(subembeddings)
        elif isinstance(value, Grid):
            return embed_grid(value)
        elif isinstance(value, Array):
            raise NotImplementedError
        elif isinstance(value, int):
            # I forget how we're handling ints/bools/colors.
            # how will we handle colors?
            raise NotImplementedError

    def embed_tensor_list(emb_list) -> Tensor
        # TODO: run through an LSTM or something?
        pass

    def embed_grid(grid: Grid) -> Tensor:
        # TODO: plug in simple CNN


    def embed_node_set(node_embeddings: List[Tensor]) -> Tensor:
        # yup, it's as simple as that.
        summed = sum(a)
        # TODO: tack on a linear layer after this.
        # TODO: should there be a nonlinearity in here?
        return self.nodeset_linear(sum(a))


    def op_logits(embedded_state):
        # TODO: should there be a nonlinearity in here?
        return self.op_choice_linear(embedded_state)

