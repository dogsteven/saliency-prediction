from torch import log, sum, mean, min, sqrt
from torch.nn import Module

__all__ = ["NormalizedScanpathSaliency", "KLDivergence", "Similarity", "CorrelationCoefficient", "PointWiseBinaryCrossEntropy", "MeanSquareError", "RootMeanSquareError"]

class NormalizedScanpathSaliency(Module):
    def forward(self, pred, y):
        pred = pred - mean(pred, 1, True)
        pred_std = sqrt(sum(pred * pred, 1, True))
        pred = pred / pred_std

        loss = sum(pred * y / sum(y, 1, True), 1)
        loss = mean(loss)
        return loss

class KLDivergence(Module):
    def __init__(self, eps = 2.2204e-16):
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

class InformationGain(Module):
    def __init__(self, eps = 2.2204e-16):
        super().__init__()
        self.eps = eps

    def forward(self, pred, y):
        loss = y * log(pred + self.eps) / sum(y, 1, True)
        loss = sum(loss, 1)
        loss = mean(loss)
        return loss

class PointWiseBinaryCrossEntropy(Module):
    def forward(self, pred, y):
        loss = y * log(pred) + (1.0 - y) * log(1.0 - pred)
        loss = mean(loss)
        return -loss

class MeanSquareError(Module):
    def forward(self, pred, y):
        diff = pred - y
        diff_square = diff * diff
        loss = mean(diff_square)
        return loss

class RootMeanSquareError(Module):
    def forward(self, pred, y):
        diff = pred - y
        diff_square = diff * diff
        loss = mean(diff_square, 1)
        loss = sqrt(loss)
        loss = mean(loss, 1)
        return loss