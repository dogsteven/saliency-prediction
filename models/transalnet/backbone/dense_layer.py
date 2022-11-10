from torch.nn import Module, BatchNorm2d, ReLU, Conv2d

__all__ = ["DenseLayer"]

class DenseLayer(Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()
        self.norm1 = BatchNorm2d(in_channels)
        self.relu1 = ReLU()
        self.conv1 = Conv2d(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False
        )

        self.norm2 = BatchNorm2d(hidden_channels)
        self.relu2 = ReLU()
        self.conv2 = Conv2d(
            in_channels = hidden_channels,
            out_channels = out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )

    def forward(self, x):
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        return x