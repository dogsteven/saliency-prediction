from torch.nn import Module

class Normalize(Module):
    def forward(self, x):
        b = x.shape[0]

        x = x - x.view(b, -1).min(1)[0].view(b, 1, 1, 1)
        x = x / (x.view(b, -1).max(1)[0].view(b, 1, 1, 1) + 1e-7)

        return x