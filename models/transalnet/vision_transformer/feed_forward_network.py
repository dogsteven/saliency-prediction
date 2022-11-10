from torch.nn import Module, Sequential, Linear, GELU, Dropout

__all__ = ["FeedForwardNetwork"]

class FeedForwardNetwork(Module):
    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            dropout_rate: float = 0.0
    ):
        super().__init__()

        self.network = Sequential(
            Linear(in_features = in_features, out_features = hidden_features),
            GELU(),
            Dropout(dropout_rate),
            Linear(in_features = hidden_features, out_features = out_features),
            Dropout(dropout_rate)
        )

    def forward(self, x):
        # shape of x: (batch, n, in_features)
        # --------------------------------------
        # shape of output: (batch, n, out_features)
        # --------------------------------------
        return self.network(x)