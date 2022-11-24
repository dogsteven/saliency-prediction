from torch.nn import Module
from torchvision.transforms.functional import resize
from .convolution_relu import Conv2dReLU

__all__ = ["GlobalAveragePooling"]

class GlobalAveragePooling(Module):
    def __init__(self):
        super().__init__()
        self.conv = Conv2dReLU(in_channels = 1280, out_channels = 256, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        b, c, h, w = x.shape

        x = x.mean(dim = (2, 3), keepdim = True)
        x = self.conv(x)
        x = resize(x, (h, w))
        return x