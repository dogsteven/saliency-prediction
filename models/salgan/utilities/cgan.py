from torch.optim import Optimizer
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

        self.automatic_optimization = False

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.generator(x)
        batch_size = x.shape[0]

        if batch_idx % 2 == 0:
            ############################
            ## optimize discriminator ##
            ############################
            d_optimizer: Optimizer = self.optimizers()[1]

            real_label = self.discriminator(x, y)
            fake_label = self.discriminator(x, pred)

            ones_label = ones((batch_size, 1)).cuda()
            zeros_label = zeros((batch_size, 1)).cuda()

            d_loss = self.d_loss_fn(real_label, ones_label) + self.d_loss_fn(fake_label, zeros_label)

            d_optimizer.zero_grad()
            self.manual_backward(d_loss)
            d_optimizer.step()

            self.log_dict({"d_loss": d_loss}, prog_bar = True)
        else:
            ########################
            ## optimize generator ##
            ########################
            g_optimizer: Optimizer = self.optimizers()[0]

            fake_label = self.discriminator(x, pred)

            ones_label = ones((batch_size, 1)).cuda()

            g_loss = self.g_loss_fn(pred, y)
            adv_loss = g_loss + self.d_loss_fn(fake_label, ones_label)

            g_optimizer.zero_grad()
            self.manual_backward(adv_loss)
            g_optimizer.step()

            self.log_dict({"g_loss": g_loss, "adv_loss": adv_loss}, prog_bar = True)