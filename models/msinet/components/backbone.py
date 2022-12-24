from torch import cat
from torch.nn import Module, MaxPool2d, ZeroPad2d
from ..utilities.convolution_relu import Conv2dReLU
from collections import OrderedDict


class VGG16Backbone(Module):
    def __init__(self):
        super().__init__()
        self.block1_conv1 = Conv2dReLU(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.block1_conv2 = Conv2dReLU(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.block1_pooling = MaxPool2d(kernel_size = 2, stride = 2)

        self.block2_conv1 = Conv2dReLU(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.block2_conv2 = Conv2dReLU(in_channels = 128, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.block2_pooling = MaxPool2d(kernel_size = 2, stride = 2)

        self.block3_conv1 = Conv2dReLU(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.block3_conv2 = Conv2dReLU(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.block3_conv3 = Conv2dReLU(in_channels = 256, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        self.block3_pooling = MaxPool2d(kernel_size = 2, stride = 2)

        self.block4_conv1 = Conv2dReLU(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.block4_conv2 = Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.block4_conv3 = Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 1)
        self.block4_padding = ZeroPad2d((0, 1, 0, 1))
        self.block4_pooling = MaxPool2d(kernel_size = 2, stride = 1)

        self.block5_conv1 = Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.block5_conv2 = Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.block5_conv3 = Conv2dReLU(in_channels = 512, out_channels = 512, kernel_size = 3, stride = 1, padding = 2, dilation = 2)
        self.block5_padding = ZeroPad2d((0, 1, 0, 1))
        self.block5_pooling = MaxPool2d(kernel_size = 2, stride = 1)

    def forward(self, x):
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)
        x = self.block1_pooling(x)

        x = self.block2_conv1(x)
        x = self.block2_conv2(x)
        x = self.block2_pooling(x)

        x = self.block3_conv1(x)
        x = self.block3_conv2(x)
        x = self.block3_conv3(x)
        x = self.block3_pooling(x)

        r1 = x

        x = self.block4_conv1(x)
        x = self.block4_conv2(x)
        x = self.block4_conv3(x)
        x = self.block4_padding(x)
        x = self.block4_pooling(x)

        r2 = x

        x = self.block5_conv1(x)
        x = self.block5_conv2(x)
        x = self.block5_conv3(x)
        x = self.block5_padding(x)
        x = self.block5_pooling(x)

        r3 = x

        return cat([r1, r2, r3], dim=1)

    def load_pretrained(self):
        from torchvision.models import vgg16

        pretrained = vgg16()

        mapping = {
            "block1_conv1": "0",
            "block1_conv2": "2",
            "block2_conv1": "5",
            "block2_conv2": "7",
            "block3_conv1": "10",
            "block3_conv2": "12",
            "block3_conv3": "14",
            "block4_conv1": "17",
            "block4_conv2": "19",
            "block4_conv3": "21",
            "block5_conv1": "24",
            "block5_conv2": "26",
            "block5_conv3": "28"
        }

        state_dict = pretrained.features.state_dict()

        mapped_weights = OrderedDict()

        for key, value in mapping.items():
            mapped_weights[f"{key}.conv.weight"] = state_dict[f"{value}.weight"]
            mapped_weights[f"{key}.conv.bias"] = state_dict[f"{value}.bias"]

        self.load_state_dict(mapped_weights)