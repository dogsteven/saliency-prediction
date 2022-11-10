from torch.nn.functional import pad
from torch.nn import Module
from torchvision.transforms import Normalize

__all__ = ["ImageNetNormalize", "PaddingByRatio"]

class ImageNetNormalize(Module):
    def __init__(self):
        super().__init__()
        self.normalize = Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225]
        )

    def forward(self, x):
        return self.normalize(x)

class PaddingByRatio(Module):
    def __init__(self, ratio: float, value: float):
        super().__init__()
        self.ratio = ratio
        self.value = value

    def forward(self, x):
        _, h, w = x.shape
        expt_h, expt_w = (h, int(h / self.ratio)) if (h / w) > self.ratio else (int(self.ratio * w), w)
        pad_h = expt_h - h
        pad_w = expt_w - w

        return pad(x, (pad_w // 2, pad_w // 2, pad_h // 2, pad_h // 2), "constant", self.value)