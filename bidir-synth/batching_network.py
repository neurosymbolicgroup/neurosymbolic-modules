import time
import math
import random
from typing import Tuple, List, Sequence, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from modules.base_modules import FC
from rl.ops.operations import Op
from rl.program_search_graph import ValueNode, ProgramSearchGraph
import modules.synth_modules


from bidir.utils import assertEqual
from bidir.primitives.types import Grid, MIN_COLOR, NUM_COLORS
import rl.ops.twenty_four_ops
import rl.ops.arc_ops
from rl.random_programs import ActionSpec, random_24_program, ProgramSpec, random_arc_small_grid_inputs_sampler, random_bidir_program

MAX_NODES = 100


def compute_out_size(in_size, model: nn.Module):
    """
    Compute output size of model for an input with size `in_size`.
    This is a utility function.
    """
    out = model(torch.Tensor(1, *in_size))  # type: ignore
    return int(np.prod(out.size()))


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
        size: int,
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


class ArgChoiceNet(nn.Module):
    def __init__(self, ops: Sequence[Op], node_dim: int, state_dim: int):
        super().__init__()
        self.ops = ops
        self.node_dim = node_dim
        self.state_dim = state_dim
        self.num_ops = len(ops)
        self.max_nodes = MAX_NODES
        self.max_arity = max(op.arity for op in ops)

    def forward(
        self,
        op_idxs: int,
        state_embeds: Tensor,
        psg_embeddings: Tensor,
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

        op_one_hot = F.one_hot(op_idxs, num_classes=self.num_ops)
        assertEqual(psg_embeddings.shape, (N, self.max_nodes, self.node_dim))
        assertEqual(op_one_hot.shape, (N, self.num_ops))
        assertEqual(state_embeds.shape, (N, self.state_dim))

        query = torch.cat([op_one_hot, state_embeds], axis=1)
        query = query.unsqueeze(1)
        query = query.repeat(1, self.max_nodes, 1)
        # query[psg_embeddings == 0.] = 0.
        in_tensor = torch.cat([query, psg_embeddings], dim=2)
        assertEqual(in_tensor.shape,
                    (N, self.max_nodes, self.state_dim + self.num_ops + self.node_dim))

        # process each node if separate elements in a batch
        arg_logits = self.args_net(in_tensor)
        assertEqual(arg_logits.shape, (N, self.max_nodes, self.max_arity))

        arg_logits2 = torch.transpose(arg_logits, 1, 2)
        assertEqual(arg_logits2.shape, (N, self.max_arity, self.max_nodes))

        if greedy:
            arg_idxs = torch.argmax(arg_logits2, dim=2)
        else:
            arg_idxs = Categorical(logits=arg_logits2).sample()
        assertEqual(arg_idxs.shape, (N, self.max_arity))

        return (arg_idxs, arg_logits2)


class DeepSetNet(nn.Module):
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
        self.max_nodes = MAX_NODES

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
        assertEqual(presum.shape, (N, self.max_nodes, self.hidden_dim))

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
                 arg_choice_net: ArgChoiceNet,
                 greedy_op: bool = False):
        super().__init__()
        self.ops = ops
        self.num_ops = len(ops)
        self.node_dim = node_dim
        self.state_dim = state_dim
        self.arg_choice_net = arg_choice_net

        # for embedding the state
        self.deepset_net = DeepSetNet(element_dim=self.node_dim,
                                      hidden_dim=self.state_dim,
                                      presum_num_layers=1,
                                      postsum_num_layers=1,
                                      set_dim=self.state_dim)
        # for choosing op.
        self.op_choice_linear = nn.Linear(self.state_dim, self.num_ops)
        self.arg_choice_net = arg_choice_net
        assert arg_choice_net.node_dim == self.node_dim, (
            'subnets all need to coordinate using the same node_dim')
        assert arg_choice_net.ops == self.ops, (
            'subnets all neet to coordinate using the same ops')

    def forward(self,
                batch_psg_embeddings: Tensor,
                greedy: bool = False) -> Tuple[Tensor, ...]:

        N = batch_psg_embeddings.shape[0]
        print(f"batch_psg_embeddings: {batch_psg_embeddings}")
        assertEqual(batch_psg_embeddings.shape, (N, MAX_NODES, self.node_dim))
        state_embeds: Tensor = self.deepset_net(batch_psg_embeddings)
        assertEqual(state_embeds.shape, (N, self.state_dim))

        op_logits = self.op_choice_linear(F.relu(state_embeds))
        assertEqual(op_logits.shape, (N, self.num_ops))

        if greedy:
            op_idxs = torch.argmax(op_logits, axis=1)
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

        return op_idxs, arg_idxs, op_logits, arg_logits  # , outs


def policy_net_24(ops: Sequence[Op], max_int: int, state_dim: int = 128) -> Tuple[PolicyNet, NodeEmbedNet]:
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
                           arg_choice_net=arg_choice_net)
    return policy_net, node_embed_net


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

        self.embedding_combiner = modules.synth_modules.DeepSetNet(
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


def policy_net_arc_v2(
    ops: Sequence[Op],
    state_dim: int = 256,
) -> Tuple[PolicyNet, NodeEmbedNet]:
    node_embed_net = ArcNodeEmbedNetGridsOnly(dim=state_dim)
    node_dim = node_embed_net.dim
    arg_choice_cls = DirectChoiceNet
    # arg_choice_cls = AutoRegressiveChoiceNet

    arg_choice_net = arg_choice_cls(ops=ops,
                                    node_dim=node_dim,
                                    state_dim=state_dim)
    policy_net = PolicyNet(ops=ops,
                           node_dim=node_dim,
                           state_dim=state_dim,
                           arg_choice_net=arg_choice_net)
    return policy_net, node_embed_net


def collate(batch: Sequence[ActionSpec], max_arity: int = 2):
    def pad_list(l, dim, pad_value=0):
        return list(l) + [pad_value]*(dim - len(l))

    return {
        'sample': batch,
        'op_class': torch.tensor([d.action.op_idx for d in batch]),
        # (batch_size, arity)
        # 'args_class':
        # torch.stack([torch.tensor(d.action.arg_idxs) for d in batch]),
        'args_class': [torch.tensor(pad_list(d.action.arg_idxs, max_arity)) for d in batch],
        'psg': [ProgramSearchGraph(d.task, d.additional_nodes) for d in batch],
    }


def train_policy_net(net, node_embed_net, train_data, val_data, lr=0.002, batch_size=128, epochs=300, print_every=1, use_cuda=False):

    max_arity = net.arg_choice_net.max_arity
    dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=lambda batch: collate(batch, max_arity))

    optimizer = optim.Adam(net.parameters(), lr=lr)
    node_embed_optimizer = optim.Adam(node_embed_net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_accuracy = -1

    use_cuda = use_cuda and torch.cuda.is_available()

    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    for epoch in range(epochs):
        start = time.time()
        total_loss = 0
        op_accuracy = 0
        args_accuracy = 0
        iters = 0

        net.train()
        node_embed_net.train()

        for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            node_embed_optimizer.zero_grad()

            op_classes = batch['op_class']
            # (batch_size, arity)
            args_classes = batch['args_class']

            batch_tensor = torch.zeros(op_classes.shape[0], MAX_NODES, node_embed_net.dim)
            for j, psg in enumerate(batch['psg']):
                assert len(psg.get_value_nodes()) <= MAX_NODES
                for k, node in enumerate(psg.get_value_nodes()):
                    batch_tensor[j, k, :] = node_embed_net(node, psg.is_grounded(node))

            if use_cuda:
                batch_tensor = batch_tensor.cuda()
                op_classes = op_classes.cuda()

            op_idxs, arg_idxs, op_logits, args_logits = net(batch_tensor, greedy=True)
            if i == 0:
                print(f"op_idxs: {op_idxs}")
                print(f"op_classes: {op_classes}")
                print(f"op_logits: {op_logits}")
            op_loss = criterion(op_logits, op_classes)

            # args_classes_tensor = torch.stack(args_classes).cuda() if use_cuda else torch.stack(args_classes)
            # arg_losses = [criterion(args_logits[:, i, :], args_classes_tensor[:, i]) for i in range(net.arg_choice_net.max_arity)]

            # for arg_logit, arg_class in zip(args_logits, args_classes):
            #     # make the class have zeros to match up with max arity
            #     arg_class = torch.cat(
            #         (arg_class, torch.zeros(net.arg_choice_net.max_arity
            #                                 - arg_class.shape[0],
            #                                 dtype=torch.long)))
            #     loss = criterion(torch.unsqueeze(arg_logit, dim=0),
            #                      torch.unsqueeze(arg_class, dim=0))
            #     arg_losses.append(loss)

            # arg_loss = sum(arg_losses) / len(arg_losses)

            # combined_loss = op_loss + arg_loss
            combined_loss = op_loss
            combined_loss.backward()
            optimizer.step()
            node_embed_optimizer.step()

            total_loss += combined_loss.sum().item()
            # only checks the first example for accuracy

            op_accuracy += (op_classes == op_idxs).float().mean().item()*100
            # args_accuracy += (args_classes_tensor == arg_idxs).float().mean().item()*100
            iters += 1

        op_accuracy = op_accuracy/iters
        # args_accuracy = args_accuracy/iters
        # val_accuracy = eval_policy_net(net, data)
        # if val_accuracy > best_val_accuracy:
        #     best_val_accuracy, best_epoch = val_accuracy, epoch
        duration = time.time() - start
        m = math.floor(duration / 60)
        s = duration - m * 60
        duration_str = f'{m}m {int(s)}s'

        if epoch % print_every == 0:
            print(
                f'Epoch {epoch} completed ({duration_str})',
                # f'op_accuracy: {op_accuracy:.2f} args_accuracy: {args_accuracy:.2f} loss: {total_loss:.2f}',
                f'op_accuracy: {op_accuracy:.2f} loss: {total_loss:.2f}',
            )
    return op_accuracy, args_accuracy  # , best_val_accuracy


def arc_training():
    data_size = 4
    val_data_size = 200
    depth = 1

    ops = rl.ops.arc_ops.FW_GRID_OPS[0:4]

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
    # for i in range(3):
    #     d = data.data[i]
    #     print(d.task, d.additional_nodes, d.action)

    # TODO: get rid of global variable
    MAX_NODES = 5

    val_programs = [sampler() for _ in range(val_data_size)]
    val_data = program_dataset(val_programs)

    net, node_embed_net = policy_net_arc_v2(ops, state_dim=5)
    op_accuracy, args_accuracy = train_policy_net(net, node_embed_net, data, val_data, print_every=10, use_cuda=True, epochs=300, batch_size=data_size, lr=0.002)
    print(op_accuracy, args_accuracy)


def twenty_four_batched_test():
    data_size = 1000
    val_data_size = 200
    depth = 1
    num_inputs = 2
    max_input_int = 15
    max_int = rl.ops.twenty_four_ops.MAX_INT

    ops = rl.ops.twenty_four_ops.SPECIAL_FORWARD_OPS[0:5]

    def sampler():
        inputs = random.sample(range(1, max_input_int + 1), k=num_inputs)
        return random_24_program(ops, inputs, depth)

    programs = [sampler() for _ in range(data_size)]
    data = program_dataset(programs)

    val_programs = [sampler() for _ in range(val_data_size)]
    val_data = program_dataset(val_programs)

    net, node_embed_net = policy_net_24(ops, max_int=max_int, state_dim=128)
    op_accuracy, args_accuracy = train_policy_net(net, node_embed_net, data, val_data, print_every=10, use_cuda=True)
    print(op_accuracy, args_accuracy)


if __name__ == '__main__':
    arc_training()
    # twenty_four_batched_test()
