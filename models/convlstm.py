
import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell with spatially preserved hidden and cell states."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # One convolution produces the input, forget, output,
        # and candidate gates.
        self.gates = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x, state):
        h_prev, c_prev = state

        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)

        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_state(self, x):
        batch, _, height, width = x.shape
        shape = (batch, self.hidden_channels, height, width)

        h = x.new_zeros(shape)
        c = x.new_zeros(shape)
        return h, c


class ConvLSTM(nn.Module):
    """Stacked ConvLSTM for one-step-ahead spatial LST prediction.

    Input:  [B, T, C, H, W]
    Output: [B, 1, H, W]
    """

    def __init__(
        self,
        input_channels=18,
        hidden_channels=64,
        kernel_size=3,
        output_channels=1,
    ):
        super().__init__()

        self.layer1 = ConvLSTMCell(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )
        self.layer2 = ConvLSTMCell(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_channels, kernel_size=1),
        )

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError(
                f"Expected [B,T,C,H,W], received {tuple(x.shape)}"
            )

        batch, steps, channels, height, width = x.shape

        if channels != 18:
            raise ValueError(f"Expected 18 input channels, got {channels}")

        h1, c1 = self.layer1.init_state(x[:, 0])
        h2, c2 = self.layer2.init_state(x[:, 0])

        for t in range(steps):
            h1, c1 = self.layer1(x[:, t], (h1, c1))
            h2, c2 = self.layer2(h1, (h2, c2))

        return self.head(h2)
