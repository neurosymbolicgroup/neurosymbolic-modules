from modules.base_modules import AllConv, FC
from bidir.utils import assertEqual
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class OldDeepSetNet(nn.Module):
    def __init__(self, element_dim, set_dim, hidden_dim=None):
        super().__init__()
        self.element_dim = element_dim
        self.set_dim = set_dim
        if not hidden_dim:
            hidden_dim = set_dim
        self.hidden_dim = hidden_dim

        self.lin1 = nn.Linear(element_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, set_dim)
        # self.finalize()

    def forward(self, node_embeddings: Tensor):
        # node_embeddings = self.tensor(node_embeddings)

        N = node_embeddings.shape[0]
        assertEqual(node_embeddings.shape[1], self.element_dim)
        out = F.relu(self.lin1(node_embeddings))
        assertEqual(out.shape, (N, self.hidden_dim))
        out = torch.sum(out, dim=0)
        assertEqual(out.shape, (self.hidden_dim, ))
        out = self.lin2(out)
        assertEqual(out.shape, (self.set_dim, ))
        return out


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

    def forward(self, node_embeddings: Tensor):
        # node_embeddings = self.tensor(node_embeddings)

        N = node_embeddings.shape[0]
        assertEqual(node_embeddings.shape[1], self.element_dim)

        presum = F.relu(self.presum_net(node_embeddings))
        assertEqual(presum.shape, (N, self.hidden_dim))

        postsum = torch.sum(presum, dim=0)
        assertEqual(postsum.shape, (self.hidden_dim, ))

        out = self.postsum_net(postsum)
        assertEqual(out.shape, (self.set_dim, ))
        return out


class PointerNet2(nn.Module):
    """
    Simpler but equivalent implementation of the pointer net.
    """
    def __init__(self, input_dim, query_dim, hidden_dim, num_hidden=1):
        super().__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden
        self.net = FC(input_dim=self.query_dim + self.input_dim,
                      output_dim=1,
                      num_hidden=self.num_hidden,
                      hidden_dim=self.hidden_dim)

        # self.finalize()

    def forward(self, inputs: Tensor, queries: Tensor):
        """
        Input:
            inputs: tensor of shape (N, input_dim)
            queries: tensor of shape (query_dim,)

        Output:
            tensor of shape (N,) with unnormalized probability of choosing each
            input.
        """
        # inputs = self.tensor(inputs)
        # queries = self.tensor(queries)
        N = inputs.shape[0]
        assertEqual(inputs.shape, (N, self.input_dim))
        assertEqual(queries.shape, (self.query_dim, ))

        # in tensor: (N, state_dim + op_dim + node_dim)
        queries = queries.repeat(N, 1)
        in_tensor = torch.cat([queries, inputs], dim=1)
        assertEqual(in_tensor.shape, (N, self.query_dim + self.input_dim))

        # each input is like a different item in the batch provided to the FC
        # net. each input gets the whole query vector as context.
        input_logits = self.net(in_tensor)
        assertEqual(input_logits.shape, (N, 1))
        input_logits = input_logits.squeeze()
        assertEqual(input_logits.shape, (N, ))

        return input_logits


class PointerNet(nn.Module):
    """
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
    def __init__(self, input_dim, query_dim, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.W2 = nn.Linear(self.query_dim, self.hidden_dim)
        self.V = nn.Linear(self.hidden_dim, 1)
        # self.finalize()

    def forward(self, inputs, queries):
        """
        Input:
            inputs: tensor of shape (N, input_dim)
            queries: tensor of shape (query_dim,)

        Output:
            tensor of shape (N,) with unnormalized probability of choosing each
            input.

        Computes additive attention identically to the pointer net paper:
        https://arxiv.org/pdf/1506.03134.pdf
        """
        # inputs = self.tensor(inputs)
        # queries = self.tensor(queries)

        N = inputs.shape[0]
        assertEqual(inputs.shape, (N, self.input_dim))
        assertEqual(queries.shape, (self.query_dim, ))

        w1_out = self.W1(inputs)
        assertEqual(w1_out.shape, (N, self.hidden_dim))
        w2_out = self.W2(queries)
        assertEqual(w2_out.shape, (self.hidden_dim, ))
        w2_repeated = w2_out.repeat(N, 1)
        assertEqual(w2_repeated.shape, (N, self.hidden_dim))
        u = self.V(torch.tanh(w2_out + w2_repeated))
        assertEqual(u.shape, (N, 1))
        u = u.squeeze(1)
        assertEqual(u.shape, (N, ))
        return u


class CNN(nn.Module):
    def __init__(self, in_channels=10, output_dim=64):
        super().__init__()

        self.all_conv = AllConv(residual_blocks=2,
                                input_filters=in_channels,
                                residual_filters=32,
                                conv_1x1s=2,
                                output_dim=output_dim,
                                conv_1x1_filters=64,
                                pooling='max')
        # self.finalize()

    def forward(self, x):
        """
        Input: tensor of shape (batch, channels, height, width)
        Output: tensor of shape (batch, output_dim)
        """
        # x = self.tensor(x)

        # (B, C, H, W) to (B, output_dim)
        x = x.to(torch.float32)
        x = self.all_conv(x)

        # test if this is actually helping.
        # return torch.rand(x.shape)
        return x


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=64):
        super().__init__()
        # see pytorch documentation for more details
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(hidden_dim, output_dim)
        # self.finalize()

    def forward(self, x):
        """
        Input: tensor of shape
            (batch, seq_len, input_dim)
        Output: tensor of shape
            (batch_size, output_dim)
        """

        # x = self.tensor(x)

        batch = x.shape[0]
        x = x.to(torch.float32)

        # (batch, seq_len, input_dim) to (batch, seq_len, hidden_dim)
        out, (last_hidden, _) = self.lstm(x)

        assert last_hidden.shape == (batch, 1, self.hidden_dim)

        last_hidden = torch.squeeze(last_hidden, 1)

        assert last_hidden.shape == (batch, self.hidden_dim)

        # TODO: nonlinearity?
        out = self.fc(last_hidden)
        assert out.shape == (batch, self.output_dim)

        # see if this is doing anything
        return torch.zeros(out.shape)
        # return out
