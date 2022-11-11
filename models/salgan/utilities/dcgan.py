from torch import cat, ones, zeros
from pytorch_lightning import LightningModule

__all__ = ["DCGAN"]

class DCGAN(LightningModule):
    def __init__(self, generator, discriminator, g_loss_fn, d_loss_fn):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.g_loss_fn = g_loss_fn
        self.d_loss_fn = d_loss_fn

        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_index):
        g_optimizer, d_optimizer = self.optimizers()

        x, y = batch
        pred = self.generator(x)

        batch_size = x.shape[0]

        ############################
        ## optimize discriminator ##
        ############################
        real_label = self.discriminator(cat([x, y], dim = 1))
        fake_label = self.discriminator(cat([x, pred], dim = 1))

        ones_label = ones((batch_size, 1))
        zeros_label = zeros((batch_size, 1))

        d_loss = self.d_loss_fn(real_label, ones_label) + self.d_loss_fn(fake_label, zeros_label)

        d_optimizer.zero_grad()
        self.manual_backward(d_loss)
        d_optimizer.step()

        ########################
        ## optimize generator ##
        ########################
        label = self.discriminator(cat([x, pred], dim = 1))

        g_loss = self.g_loss_fn(pred, y) + self.d_loss_fn(label, ones_label)

        g_optimizer.zero_grad()
        self.manual_backward(g_loss)
        g_optimizer.step()

        self.log_dict({"g_loss": g_loss, "d_loss": d_loss}, prog_bar = True)