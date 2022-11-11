from torch.nn import Module

__all__ = ["FusionGate"]

class FusionGate(Module):
    def forward(self, features, contextual_vectors):
        c = contextual_vectors.shape[1]
        return features * contextual_vectors.reshape(-1, c, 1, 1)