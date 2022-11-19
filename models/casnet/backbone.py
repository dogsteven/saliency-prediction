from torch.nn import Module, MaxPool2d, Sequential
from .utilities.convolution_relu import Conv2dReLU
from collections import OrderedDict

__all__ = ["VGG16Backbone"]

from collections import OrderedDict

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

    def load_pretrained(self):
        from torchvision.models import vgg16

        pretrained = vgg16()

        mapping = {
            "0": "0",
            "1": "2",
            "3": "5",
            "4": "7",
            "6": "10",
            "7": "12",
            "8": "14",
            "10": "17",
            "11": "19",
            "12": "21",
            "14": "24",
            "15": "26",
            "16": "28"
        }

        state_dict = pretrained.features.state_dict()
        mapped_weights = OrderedDict()

        for key, value in mapping.items():
            mapped_weights[f"{key}.conv.weight"] = state_dict[f"{value}.weight"]
            mapped_weights[f"{key}.conv.bias"] = state_dict[f"{value}.bias"]

        self.network.load_state_dict(mapped_weights)