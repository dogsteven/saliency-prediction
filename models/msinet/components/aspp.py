from torch import cat
from torch.nn import Module
from torchvision.transforms.functional import resize
from ..utilities.convolution_relu import Conv2dReLU
from ..utilities.global_average_pooling import GlobalAveragePooling

__all__ = ["ASPP"]

class ASPP(Module):
    def __init__(self):
        super().__init__()
        self.branch1 = Conv2dReLU(in_channels = 1280, out_channels = 256, kernel_size = 1, stride = 1, padding = 0)
        self.branch2 = Conv2dReLU(in_channels = 1280, out_channels = 256, kernel_size = 3, stride = 1, padding = 4, dilation = 4)
        self.branch3 = Conv2dReLU(in_channels = 1280, out_channels = 256, kernel_size = 3, stride = 1, padding = 8, dilation = 8)
        self.branch4 = Conv2dReLU(in_channels = 1280, out_channels = 256, kernel_size = 3, stride = 1, padding = 12, dilation = 12)
        self.branch5 = GlobalAveragePooling()

        self.combine = Conv2dReLU(in_channels = 1280, out_channels = 256, kernel_size = 1, stride = 1, padding = 0)


    def forward(self, x):
        r1 = self.branch1(x)
        r2 = self.branch2(x)
        r3 = self.branch3(x)
        r4 = self.branch4(x)
        r5 = self.branch5(x)

        r = cat([r1, r2, r3, r4, r5], dim = 1)
        r = self.combine(r)

        return