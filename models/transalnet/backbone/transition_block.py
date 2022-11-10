from torch.nn import Module, BatchNorm2d, ReLU, Conv2d, AvgPool2d

__all__ = ["TransitionBlock"]

class TransitionBlock(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = BatchNorm2d(in_channels)
        self.relu = ReLU()
        self.conv = Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = False
        )
        self.pool = AvgPool2d(
            kernel_size = 2,
            stride = 2,
            padding = 0
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.pool(x)

        return x