## Measurement

The `measure.measures` module provides some distributed-based saliency loss/metric functions, including KL Divergence, Similarity and Pearson Correlation Coefficient. These measure functions require inputs of the form `(pred, y)` where `pred` and `y` are matrices satisfying the following condiitons:
1. They are of shape `[B, S]` where `B` is the batch size and `N` is the total spatial size.
2. Each vector in a batch is probabilitic.

A saliency model usually return a batch of saliency maps of shape `[B, 1, H, W]` where `B` is the batch size, `H` and `W` are the spatial dimensions. To use those measure functions, first we need to flatten the tensors to shape `[B, H * W]` then normalize the `1`-th coordinate to probabilitic. The `measure.transforms` module provides utility functions for preprocessing the tensors before passing them to a measure function.

Example:

```python
class TranSalNetLoss(Module):
    def __init__(self):
        super().__init__()
        from .measure.transforms import *
        from .measure.measures import *
        
        component = ChainTransform([
            BatchFlatten(),
            NormalizeToProbabilitic()
        ])
        
        self.transform = ComponentWiseTransform(component)
        self.kld = KLDivergence()
        self.sim = Similarity()
        self.pcc = CorrelationCoefficient()

    def forward(self, pred, y):
        pred, y = self.transform(pred, y)
        return 10.0 * self.kld(pred, y) - self.sim(pred, y) - self.pcc(pred, y)
```

## Training

The `training` module provides a `train` function for training a `Pytorch Lightning` module:
1. The `model` parameter requires a `LightningModule` model.
2. The `name` parameter requires the name of the model.
3. The `train_dataset` parameter requires a training `SaliencyMapDataset`.
4. The `val_dataset` parameter requires a validation `SaliencyMapDataset`.
5. The `gpus` parameter requires the number of gpus used during the training step.
6. The `batch_size` parameter requires the batch size.
7. The `max_epochs` parameter requires the maximum epochs of the training step.

```python
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiplicativeLR
from pytorch_lightning import LightningModule
from torchvision.transforms import ToTensor, Compose, Resize
from dataset.salicon import SALICONSaliencyDataset
from models.transalnet.model import TranSalNet
from training import train
from image_preprocessing import ImageNetNormalize

image_transform = Compose([
    ToTensor(),
    Resize((288, 384)),
    ImageNetNormalize()
])

saliency_map_transform = Compose([
    ToTensor(),
    Resize((288, 384))
])

train_dataset = SALICONSaliencyDataset(
    image_directory = "./salicon/images/train",
    image_transform = image_transform,
    saliency_map_directory = "./salicon/maps/train",
    saliency_map_transform = saliency_map_transform
)

val_dataset = SALICONSaliencyDataset(
    image_directory = "./salicon/images/val",
    image_transform = image_transform,
    saliency_map_directory = "./salicon/maps/val",
    saliency_map_transform = saliency_map_transform
)

class TranSalNetLoss(Module):
    def __init__(self):
        super().__init__()
        from measure.transforms import *
        from measure.measures import *
        
        component = ChainTransform([
            BatchFlatten(),
            NormalizeToProbabilitic()
        ])
        
        self.transform = ComponentWiseTransform(component)
        self.kld = KLDivergence()
        self.sim = Similarity()
        self.pcc = CorrelationCoefficient()

    def forward(self, pred, y):
        pred, y = self.transform(pred, y)
        return 10.0 * self.kld(pred, y) - self.sim(pred, y) - self.pcc(pred, y)

class LightningWrapper(LightningModule):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_index):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        self.log("val_loss", loss, sync_dist = True)
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr = 1e-4)
        lr_scheduler = {
            "scheduler": MultiplicativeLR(optimizer, lambda epoch: 0.1 if epoch % 3 == 0 else 1.0),
            "name": "transalnet-lr-scheduler"
        }
        return [optimizer], [lr_scheduler]

model = TranSalNet()
loss_fn = TranSalNetLoss()
wrapper = LightningWrapper(model, loss_fn)

train(wrapper, "transalnet", train_dataset, val_dataset, gpus = 2, batch_size = 8, max_epochs = 30)
```

## Models

The `models` module provides some saliency models, including `TranSalNet-Dense`, `MSINet`, `CASNet` and `SalGAN`.