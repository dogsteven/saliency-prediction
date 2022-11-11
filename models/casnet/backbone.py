from torch.nn import Module, MaxPool2d, Sequential
from .utilities.convolution_relu import Conv2dReLU

__all__ = ["VGG16Backbone"]

class VGG16Backbone(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(
            Conv2dReLU(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            MaxPool2d(kernel_size = 2, stride = 2),
            Conv2dReLU(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            MaxPool2d(kernel_size = 2, stride = 2),
            Conv2dReLU(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            MaxPool2d(kernel_size = 2, stride = 2),
            Conv2dReLU(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            MaxPool2d(kernel_size = 2, stride = 2),
            Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            MaxPool2d(kernel_size = 2, stride = 2)
        )

    def forward(self, x):
        return self.network(x)