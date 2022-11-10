from torch import cat
from torch.nn import Module, ModuleList
from .dense_layer import DenseLayer

__all__ = ["DenseBlock"]

class DenseBlock(Module):
    def __init__(self, in_channels, n_layers, grownth_rate, hidden_channels):
        super().__init__()
        layers = []

        for i in range(n_layers):
            layers.append(
                DenseLayer(
                    in_channels = in_channels + i * grownth_rate,
                    out_channels = grownth_rate,
                    hidden_channels = hidden_channels
                )
            )

        self.layers = ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            y = layer(x)
            x = cat([x, y], 1)

        return x

    def load_pretrained(self, pretrained_block):
        for layer, pretrained_layer in zip(self.layers, pretrained_block.children()):
            layer.load_state_dict(pretrained_layer.state_dict())