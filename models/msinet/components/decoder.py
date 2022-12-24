from torch.nn import Module, UpsamplingBilinear2d, Conv2d
from ..utilities.convolution_relu import Conv2dReLU

__all__ = ["Decoder"]

class Decoder(Module):
    def __init__(self):
        super().__init__()
        self.block1_upsampling = UpsamplingBilinear2d(scale_factor = 2.0)
        self.block1_conv = Conv2dReLU(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)

        self.block2_upsampling = UpsamplingBilinear2d(scale_factor = 2.0)
        self.block2_conv = Conv2dReLU(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)

        self.block3_upsampling = UpsamplingBilinear2d(scale_factor = 2.0)
        self.block3_conv = Conv2dReLU(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        self.block4_conv = Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        x = self.block1_upsampling(x)
        x = self.block1_conv(x)

        x = self.block2_upsampling(x)
        x = self.block2_conv(x)

        x = self.block3_upsampling(x)
        x = self.block3_conv(x)

        x = self.block4_conv(x)

        return x