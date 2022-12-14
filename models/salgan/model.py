from torch.optim import Adagrad
from .utilities.cgan import ConditionalGAN

__all__ = ["SalGAN"]

class SalGAN(ConditionalGAN):
    def configure_optimizers(self):
        g_optimizer = Adagrad(self.generator.parameters(), lr = 3e-4, weight_decay = 1e-4)
        d_optimizer = Adagrad(self.discriminator.parameters(), lr = 3e-4, weight_decay = 1e-4)

        return g_optimizer, d_optimizer