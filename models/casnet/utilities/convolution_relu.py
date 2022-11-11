from torch.nn import Module, Conv2d, ReLU
from torch.nn.init import xavier_uniform_ as xavier_uniform
from torch.nn.init import uniform_ as uniform

__all__ = ["Conv2dReLU"]

class Conv2dReLU(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, dilation: int = 1):
        super().__init__()
        self.conv = Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation
        )
        self.relu = ReLU()

        xavier_uniform(self.conv.weight)
        uniform(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x