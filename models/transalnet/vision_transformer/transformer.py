from torch.nn import Module, LayerNorm, Sequential
from .multi_head_attention import MultiHeadAttention
from .feed_forward_network import FeedForwardNetwork

__all__ = ["TransformerEncoder"]

class TransformerEncoderLayer(Module):
    def __init__(
        self,
        n_heads: int,
        d_model: int,
        ffn_hidden_features: int,
        attention_dropout_rate: float = 0.0,
        ffn_dropout_rate: float = 0.0,
        bias: bool = False
    ):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_qkv = (d_model, d_model, d_model),
            n_heads = n_heads,
            d_model = d_model,
            d_output = d_model,
            dropout_rate = attention_dropout_rate,
            bias = bias
        )
        self.ffn = FeedForwardNetwork(
            in_features = d_model,
            hidden_features = ffn_hidden_features,
            out_features = d_model,
            dropout_rate = ffn_dropout_rate
        )
        self.attention_layer_norm = LayerNorm(d_model, eps = 1e-6)
        self.ffn_layer_norm = LayerNorm(d_model, eps = 1e-6)

    def forward(self, x):
        # shape of x: (batch, n, d_model)
        # --------------------------------------
        # shape of output: (batch, n, d_model)
        # --------------------------------------

        h = x  # shape (batch, n, d_model)
        x = self.attention_layer_norm(x)  # shape (batch, n, d_model)
        x = self.attention(x, x, x)  # shape (batch, n, d_model)
        x = x + h  # shape (batch, n, d_model)

        h = x  # shape (batch, n, d_model)
        x = self.ffn_layer_norm(x)  # shape (batch, n, d_model)
        x = self.ffn(x)  # shape (batch, n, d_model)
        x = x + h  # shape (batch, n, d_model)

        return x

    def forward_for_visualization(self, x):
        h = x  # shape (batch, n, d_model)
        x = self.attention_layer_norm(x)  # shape (batch, n, d_model)
        x, visualization_result = self.attention.forward_for_visualization(x, x, x)  # shape (batch, n, d_model)
        x = x + h  # shape (batch, n, d_model)

        h = x  # shape (batch, n, d_model)
        x = self.ffn_layer_norm(x)  # shape (batch, n, d_model)
        x = self.ffn.forward_for_visualization(x)  # shape (batch, n, d_model)
        x = x + h  # shape (batch, n, d_model)

        return x, visualization_result


class TransformerEncoder(Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            d_model: int,
            ffn_hidden_features: int,
            attention_dropout_rate: float = 0.0,
            ffn_dropout_rate: float = 0.0,
    ):
        super().__init__()

        self.layers = Sequential(*[
            TransformerEncoderLayer(
                n_heads = n_heads,
                d_model = d_model,
                ffn_hidden_features = ffn_hidden_features,
                attention_dropout_rate = attention_dropout_rate,
                ffn_dropout_rate = ffn_dropout_rate
            )
            for _ in range(n_layers)
        ])

        self.encoder_layer_norm = LayerNorm(d_model, eps = 1e-6)

    def forward(self, x):
        # shape of x: (batch, n, d_model)
        # --------------------------------------
        # shape of output: (batch, n, d_model)
        # --------------------------------------

        x = self.layers(x)  # shape (batch, n, d_model)
        x = self.encoder_layer_norm(x)  # shape (batch, n, d_model)
        return x

    def forward_for_visualization(self, x):
        visualization_results = []
        for layer in self.layers:
            x, visualization_result = layer.forward_for_visualization(x)
            visualization_results.append(visualization_result)
        x = self.encoder_layer_norm(x)
        return x, visualization_results