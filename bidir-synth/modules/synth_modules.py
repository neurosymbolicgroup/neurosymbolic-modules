from modules.base_modules import AllConv
import torch
import torch.nn as nn
import torch.nn.functional as F


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


    def forward(self, x):
        """
        Input: tensor of shape (batch, channels, height, width)
        Output: tensor of shape (batch, output_dim)
        """

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



    def forward(self, x):
        """
        Input: tensor of shape
            (batch, seq_len, input_dim)
        Output: tensor of shape
            (batch_size, output_dim)
        """

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

        return out

