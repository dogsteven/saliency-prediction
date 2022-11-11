## Defining a `Dataset` for saliency models

By implementing `SaliencyMapDataset`, we shall create a `Dataset` class for saliency models:
1. The `image_transform` parameter of the initializer requires a `Callable` which receives a `PIL.Image` and returns a `torch.Tensor` of `3` channels.
2. The `ground_truth_transform` parameter of the initializer requires a `Callable` which receives a `PIL.Image` and returns a `torch.Tensor` of `1` channel.
3. The method `get_image_path(self, index: int) -> str` returns the path of the `index`-th image of the dataset.
4. The method `get_ground_truth_path(self, index: int) -> str` returns the path of the `index`-th ground truth of the dataset.

Another way is to implement `SaliencyMapDirectoryDataset`: Assume that the images of the dataset are stored in the same directory, and the ground truths of the dataset are stored in the same diretory too.
1. The `image_directory` parameter of the initializer is the path to the images directory.
2. The `ground_truth_directory` parameter of the initializer is the path to the ground truths directory.
3. The `image_transform` and `ground_truth_transform` parameters are the same as the `SaliencyMapDataset`'s initializer.
4. The method `get_image_subpath(self, index: int) -> str` returns the subpath of the `index`-th image of the dataset with respect to `image_directory` directory.
5. The method `get_ground_truth_subpath(self, index: int) -> str` returns the subpath of the `index`-th ground truth of the dataset with respect to `ground_truth_directory` directory.

For detailed implementations, please check `dataset.cat2000` and `dataset.salicon` modules.

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
from dataset.salicon import SALICONSaliencyMapDataset
from models.transalnet.model import TranSalNet
from training import train
from image_preprocessing import ImageNetNormalize

image_transform = Compose([
    ToTensor(),
    Resize((288, 384)),
    ImageNetNormalize()
])

ground_truth_transform = Compose([
    ToTensor(),
    Resize((288, 384))
])

train_dataset = SALICONSaliencyMapDataset(
    image_directory = "./salicon/images/train",
    ground_truth_directory = "./salicon/maps/train",
    image_transform = image_transform,
    ground_truth_transform = ground_truth_transform
)

val_dataset = SALICONSaliencyMapDataset(
    image_directory = "./salicon/images/val",
    ground_truth_directory = "./salicon/maps/val",
    image_transform = image_transform,
    ground_truth_transform = ground_truth_transform
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