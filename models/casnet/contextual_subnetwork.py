from torch.nn import Module, MaxPool2d, Linear, Flatten, Sequential

__all__ = ["ContextualSubnetwork"]

class ContextualSubnetwork(Module):
    def __init__(self):
        super().__init__()
        self.network = Sequential(
            MaxPool2d(kernel_size = 2, stride = 2),
            Flatten(),
            Linear(in_features = 9 * 12 * 1024, out_features = 1024)
        )

    def forward(self, x):
        return self.network(x)