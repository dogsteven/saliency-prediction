from torch import zeros
from torch.nn import Module, Parameter

__all__ = ["PositionalEncoding"]

class PositionalEncoding(Module):
    def __init__(self, shape):
        super().__init__()
        self.position = Parameter(zeros(shape))

    def forward(self, x):
        return x + self.position