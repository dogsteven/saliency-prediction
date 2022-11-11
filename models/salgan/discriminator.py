from torch.nn import Module, Tanh, Sigmoid, MaxPool2d, Linear, Sequential
from .utilities.convolution_relu import Conv2dReLU

__all__ = ["Discriminator"]

class Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(
            Conv2dReLU(in_channels = 4, out_channels = 3, kernel_size = 1, stride = 1, padding = 0),
            Conv2dReLU(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            MaxPool2d(kernel_size = 2, padding = 2),
            Conv2dReLU(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            MaxPool2d(kernel_size = 2, padding = 2),
            Conv2dReLU(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            Conv2dReLU(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            MaxPool2d(kernel_size = 2, padding = 2),
            Linear(in_features = 24 * 32 * 64, out_features = 100),
            Tanh(),
            Linear(in_features = 100, out_features = 2),
            Tanh(),
            Linear(in_features = 2, out_features = 1),
            Sigmoid()
        )

    def forward(self, x):
        return self.network(x)