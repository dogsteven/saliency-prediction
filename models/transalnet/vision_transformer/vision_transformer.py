from torch.nn import Module
from .patch_embedding import PatchEmbedding
from .transformer import TransformerEncoder

__all__ = ["VisionTransformer"]

def convert_to_object(**kwargs):
    class Object:
        pass

    obj = Object()

    for key, value in kwargs.items():
        setattr(obj, key, value)

    return obj


class VisionTransformer(Module):
    def __init__(self, **config):
        config = convert_to_object(**config)
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            in_channels = config.in_channels,
            output_dim = config.d_model,
            spatial_shape = config.spatial_shape
        )

        self.transformer_encoder = TransformerEncoder(
            n_layers = config.n_layers,
            n_heads = config.n_heads,
            d_model = config.d_model,
            ffn_hidden_features = config.ffn_hidden_features,
            attention_dropout_rate = config.attention_dropout_rate,
            ffn_dropout_rate = config.ffn_dropout_rate
        )

    def forward(self, features):
        # shape of features: (batch, in_channels, h, w)
        # --------------------------------------
        # shape of output: (batch, d_model, h, w)
        # --------------------------------------

        b, _, h, w = features.shape
        features = self.patch_embedding(features)  # shape (batch, h * w, d_model)
        features = self.transformer_encoder(features)
        features = features.transpose(1, 2)
        features = features.reshape(b, -1, h, w)

        return features

    def forward_for_visualization(self, features):
        b, _, h, w = features.shape
        features = self.patch_embedding.forward_for_visualization(features)
        features = self.transformer_encoder.forward_for_visualization(features)
        features = features.transpose(1, 2)
        features = features.reshape(b, -1, h, w)

        return features