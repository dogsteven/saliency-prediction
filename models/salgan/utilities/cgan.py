from torch import cat, ones, zeros
from pytorch_lightning import LightningModule

__all__ = ["ConditionalGAN"]

class ConditionalGAN(LightningModule):
    def __init__(self, generator, discriminator, g_loss_fn, d_loss_fn):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        x, y = batch
        pred = self.generator(x)
        batch_size = x.shape[0]

        if optimizer_idx == 0:
            true_label = self.discriminator(x, y)
            fake_label = self.discriminator(x, pred)

            ones_label = ones((batch_size, 1)).cuda()
            zeros_label = zeros((batch_size, 1)).cuda()

            d_loss = self.d_loss_fn(true_label, ones_label) + self.d_loss_fn(fake_label, zeros_label)

            return {
                "loss": d_loss,
                "progress_bar": {
                    "d_loss": d_loss
                },
                "log": {
                    "d_loss": d_loss
                }
            }
        elif optimizer_idx == 1:
            fake_label = self.discriminator(x, pred)
            ones_label = ones((batch_size, 1)).cuda()

            g_loss = self.g_loss_fn(pred, y) + self.d_loss_fn(fake_label, ones_label)
            return {
                "loss": g_loss,
                "progress_bar": {
                    "g_loss": g_loss
                },
                "log": {
                    "g_loss": g_loss
                }
            }
