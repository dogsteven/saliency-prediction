from torch import sum
from torch.nn import Module, Sequential

__all__ = ["BatchFlatten", "NormalizeToProbabilitic", "ChainTransform", "ComponentWiseTransform", "ComponentCombineTransform", "GetComponent"]

class ChainTransform(Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = Sequential(*transforms)

    def forward(self, x):
        return self.transforms(x)

class ComponentWiseTransform(Module):
    def __init__(self, component):
        super().__init__()
        self.component = component

    def forward(self, pred, y):
        return self.component(pred), self.component(y)

class ComponentCombineTransform(Module):
    def __int__(self, pred_transform, y_transform):
        super().__init__()
        self.pred_transform = pred_transform
        self.y_transform = y_transform

    def forward(self, pred, y):
        return self.pred_transform(pred), self.y_transform(y)

class GetComponent(Module):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, x):
        return x[self.index]

class BatchFlatten(Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

class NormalizeToProbabilitic(Module):
    def __init__(self, eps = 2.2204e-16):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / (sum(x, 1, True) + self.eps)