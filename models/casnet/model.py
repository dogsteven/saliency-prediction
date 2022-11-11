from torch import cat
from torch.nn import Module, Conv2d, Sigmoid
from torch.nn.functional import interpolate
from .backbone import VGG16Backbone
from .contextual_subnetwork import ContextualSubnetwork
from .fusion_gate import FusionGate

class CASNet(Module):
    def __init__(self):
        super().__init__()
        self.fine_backbone = VGG16Backbone()
        self.coarse_backbone = VGG16Backbone()
        self.contextual_subnetwork = ContextualSubnetwork()
        self.fusion_gate = FusionGate()
        self.conv = Conv2d(in_channels = 1024, out_channels = 1, kernel_size = 1, stride = 1, padding = 0)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        fine_features = self.fine_backbone(x)
        coarse_features = self.coarse_backbone(interpolate(x, size = (300, 400), mode = "bilinear"))

        features = cat([fine_features, interpolate(coarse_features, size = (18, 25), mode = "bilinear")], dim = 1)
        contextual_vectors = self.contextual_subnetwork(features)

        features = self.fusion_gate(features, contextual_vectors)

        maps = self.conv(features)
        maps = self.sigmoid(maps)

        return maps
