from torch import log, sum, mean, min, sqrt
from torch.nn import Module

__all__ = ["KLDivergence", "Similarity", "CorrelationCoefficient"]

class KLDivergence(Module):
    def __init__(self, eps = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, pred, y):
        loss = y * log(self.eps + y / (pred + self.eps))
        loss = sum(loss, 1)
        loss = mean(loss)
        return loss

class Similarity(Module):
    def forward(self, pred, y):
        loss = min(pred, y)
        loss = sum(loss, 1)
        loss = mean(loss)
        return loss

class CorrelationCoefficient(Module):
    def forward(self, pred, y):
        pred = pred - mean(pred, 1, True)
        y = y - mean(y, 1, True)

        pred_var = sqrt(sum(pred * pred, 1))
        y_var = sqrt(sum(y * y, 1))
        cov = sum(pred * y, 1)

        cc = cov / (pred_var * y_var)
        cc = mean(cc)
        return cc

class PointWiseBinaryCrossEntropy(Module):
    def forward(self, pred, y):
        loss = y * log(pred) + (1.0 - y) * log(1.0 - pred)
        loss = sum(loss, 1)
        loss = mean(loss)
        return loss