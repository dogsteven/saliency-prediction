from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, MaxPool2d
from .dense_block import DenseBlock
from .transition_block import TransitionBlock

__all__ = ["DenseNet161Backbone"]

class DenseNet161Backbone(Module):
    def __init__(self):
        super().__init__()
        self.conv0 = Conv2d(
            in_channels = 3,
            out_channels = 96,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            bias = False
        )
        self.norm0 = BatchNorm2d(96)
        self.relu0 = ReLU()
        self.pool0 = MaxPool2d(
            kernel_size = 3,
            stride = 2,
            padding = 1
        )

        self.denseblock1 = DenseBlock(in_channels = 96, n_layers = 6, grownth_rate = 48, hidden_channels = 192)
        self.transition1 = TransitionBlock(in_channels = 384, out_channels = 192)
        self.denseblock2 = DenseBlock(in_channels = 192, n_layers = 12, grownth_rate = 48, hidden_channels = 192)
        self.transition2 = TransitionBlock(in_channels = 768, out_channels = 384)
        self.denseblock3 = DenseBlock(in_channels = 384, n_layers = 36, grownth_rate = 48, hidden_channels = 192)
        self.transition3 = TransitionBlock(in_channels = 2112, out_channels = 1056)
        self.denseblock4 = DenseBlock(in_channels = 1056, n_layers = 24, grownth_rate = 48, hidden_channels = 192)

    def forward(self, x):
        x = self.conv0(x)
        x = self.norm0(x)
        x = self.relu0(x)
        x = self.pool0(x)
        x = self.denseblock1(x)
        x = self.transition1(x)
        x = self.denseblock2(x)
        r1 = x
        x = self.transition2(x)
        x = self.denseblock3(x)
        r2 = x
        x = self.transition3(x)
        x = self.denseblock4(x)
        r3 = x
        return r1, r2, r3

    def load_pretrained(self):
        from torchvision.models import densenet161

        features = densenet161(pretrained = True).features
        self.conv0.load_state_dict(features.conv0.state_dict())
        self.norm0.load_state_dict(features.norm0.state_dict())
        self.denseblock1.load_pretrained(features.denseblock1)
        self.transition1.load_state_dict(features.transition1.state_dict())
        self.denseblock2.load_pretrained(features.denseblock2)
        self.transition2.load_state_dict(features.transition2.state_dict())
        self.denseblock3.load_pretrained(features.denseblock3)
        self.transition3.load_state_dict(features.transition3.state_dict())
        self.denseblock4.load_pretrained(features.denseblock4)