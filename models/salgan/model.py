from torch.optim import Adagrad
from .utilities.dcgan import DCGAN

__all__ = ["SalGAN"]

class SalGAN(DCGAN):
    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        g_optimizer = Adagrad(self.generator.parameters(), lr = 3e-10, weight_decay = 1e-4)
        d_optimizer = Adagrad(self.discriminator.parameters(), lr = 3e-10, weight_decay = 1e-4)

        return d_optimizer, g_optimizer