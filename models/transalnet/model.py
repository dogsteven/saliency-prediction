from torch.nn import Module, Conv2d, BatchNorm2d, ReLU, Sequential, Upsample, Sigmoid
from .backbone.densenet_backbone import DenseNet161Backbone
from .vision_transformer.vision_transformer import VisionTransformer

__all__ = ["TranSalNet"]

class TranSalNet(Module):
    def __init__(self):
        super().__init__()
        self.backbone = DenseNet161Backbone()

        self.vit1 = VisionTransformer(
            in_channels = 768,
            spatial_shape = (36, 48),
            d_model = 512,
            ffn_hidden_features = 512 * 4,
            n_heads = 8,
            n_layers = 2,
            attention_dropout_rate = 0.0,
            ffn_dropout_rate = 0.0,
            bias = False
        )

        self.vit2 = VisionTransformer(
            in_channels = 2112,
            spatial_shape = (18, 24),
            d_model = 768,
            ffn_hidden_features = 512 * 4,
            n_heads = 12,
            n_layers = 2,
            attention_dropout_rate = 0.0,
            ffn_dropout_rate = 0.0,
            bias = False
        )

        self.vit3 = VisionTransformer(
            in_channels = 2208,
            spatial_shape = (9, 12),
            d_model = 768,
            ffn_hidden_features = 512 * 4,
            n_heads = 12,
            n_layers = 2,
            attention_dropout_rate = 0.0,
            ffn_dropout_rate = 0.0,
            bias = False
        )

        self.block1 = Sequential(
            Conv2d(in_channels = 768, out_channels = 768, kernel_size = 3, stride = 1, padding=1),
            BatchNorm2d(768),
            ReLU(True),
            Upsample(scale_factor = 2, mode = "nearest")
        )

        self.block2 = Sequential(
            Conv2d(in_channels = 768, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(512),
            ReLU(True),
            Upsample(scale_factor = 2, mode = "nearest")
        )

        self.block3 = Sequential(
            Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(256),
            ReLU(True),
            Upsample(scale_factor = 2, mode = "nearest")
        )

        self.block4 = Sequential(
            Conv2d(in_channels = 256, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(128),
            ReLU(True),
            Upsample(scale_factor = 2, mode = "nearest")
        )

        self.block5 = Sequential(
            Conv2d(in_channels = 128, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(64),
            ReLU(True),
            Upsample(scale_factor = 2, mode = "nearest")
        )

        self.block6 = Sequential(
            Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm2d(32),
            ReLU(True)
        )

        self.block7 = Sequential(
            Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 1, padding = 1),
            Sigmoid()
        )

        self.backbone.load_pretrained()

    def forward(self, x):
        x1, x2, x3 = self.backbone(x)

        x3 = self.vit3(x3)
        x3 = self.block1(x3)

        x2a = self.vit2(x2)
        x2 = x3 * x2a
        x2 = self.block2(x2)

        x1a = self.vit1(x1)
        x1 = x2 * x1a
        x1 = self.block3(x1)

        x = self.block4(x1)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        return x