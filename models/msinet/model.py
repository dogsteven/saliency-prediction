from torch.nn import Module
from .components.backbone import VGG16Backbone
from .components.aspp import ASPP
from .components.decoder import Decoder
from .components.normalize import Normalize


class MSINet(Module):
    def __init__(self):
        super().__init__()
        self.backbone = VGG16Backbone()
        self.aspp = ASPP()
        self.decoder = Decoder()
        self.normalize = Normalize()

    def forward(self, x):
        x = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x)
        x = self.normalize(x)
        return x