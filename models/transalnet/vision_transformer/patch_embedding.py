from torch.nn import Module, Conv2d
from .positional_encoding import PositionalEncoding
from typing import Tuple

__all__ = ["PatchEmbedding"]

class PatchEmbedding(Module):
    def __init__(
            self,
            in_channels: int,
            output_dim: int,
            spatial_shape: Tuple[int, int]
    ):
        super().__init__()

        self.projection = Conv2d(
            in_channels = in_channels,
            out_channels = output_dim,
            kernel_size = 1,
            stride = 1,
            padding = 0
        )

        h, w = spatial_shape
        self.position_encoding = PositionalEncoding((1, h * w, output_dim))

    def forward(self, features):
        # shape of features: (batch, in_channels, h, w)
        # --------------------------------------
        # shape of output: (batch, h * w, output_dim)
        # --------------------------------------

        features = self.projection(features)  # shape (batch, output_dim, h, w)
        features = features.flatten(2)  # shape (batch, output_dim, h * w)
        features = features.transpose(1, 2)  # shape (batch, h * w, output_dim)
        features = self.position_encoding(features)  # shape (batch, h * w, output_dim)
        return features