## Dataset

A `Dataset` object can be considered as an ordered collection of objects. Most of datasets used in supervised learning are collections of `(input, ground truth)` pairs. In `torch`, every time we want a `Dataset` object for a specific model, we must define the dataset from scratch by implementing `Dataset` class, which is too boilerplated.

A dataset for a saliency prediction model is usually a collection of either `(image, image)` or `(image, (image, image))` pairs. Therefore, instead of implementing `Dataset` class every time we want a dataset, we should implement a sort of dataset just for images only, together with some functionalities for combining those datasets.   

The `dataset` module provides `ImageDataset` and `ImageDirectoryDataset` classes for representating datasets of images. To see how these classes work, plese check the implementation.

The `dataset` module also provides `ZippedDataset` for zipping datasets: Let `d1` and `d2` be two datasets, and let `d = ZippedDataset(d1, d2)`, the object `d[index]` is just `(d1[index], d2[index])` pair.

Example: Suppose the SALICON dataset includes 3 directories:
1. `./salicon/images` contains 3 subdirectories `train`, `val` and `test`, each subdirectory contains images.
2. `./salicon/maps` contains 2 subdirectories `train` and `val`, each subdirectory contains saliency maps.
3. `./salicon/fixations` contains 2 subdirectories `train` and `val`, each subdirectory contains fixation maps.

Suppose that we want a training dataset whose elements are `(image, (saliency map, fixation map))` pairs. First, we create a dataset `training_image_dataset` of images in the `./salicon/images/train` directory. Next, we create a dataset `training_saliency_map_dataset` of images in the `./salicon/maps/train` directory. Next, we create a dataset `training_fixation_map_dataset` of images in the `./salicon/fixations/train` directory. Finally, we combine them via `ZippedDataset`.
```python
from dataset.dataset import ZippedDataset
from dataset.image_dataset import ImageDirectorySourceDataset

training_image_dataset = ImageDirectorySourceDataset(
    directory = "./salicon/images/train",
    trasform = image_transform
)

training_saliency_map_dataset = ImageDirectorySourceDataset(
    directory = "./salicon/maps/train",
    transform = saliency_map_transform
)

training_fixation_map_dataset = ImageDirectorySourceDataset(
    directory = "./salicon/fixations/train",
    transform = fixation_map_transform
)

salicon_training_dataset = ZippedDataset(
    training_image_dataset,
    ZippedDataset(
        training_saliency_map_dataset,
        training_fixation_map_dataset
    )
)
```

## Measurement

The `measure` module provides some metric functions for saliency models, such as KLD, NSS, LCC, SIM, MSE, RMSE, ... together with functionalities for prepreprocessing the inputs before passing them to the metric functions.  

Example:

```python
from torch.nn import Module
from torch.nn.functional import interpolate

class TranSalNetLoss(Module):
    def __init__(self):
        super().__init__()
        from measure.measures import *
        from measure.transforms import *
        
        self.batch_flatten = BatchFlatten()
        self.normalize_to_probabilitic = NormalizeToProbabilitic()
        
        self.kld = KLDivergence(eps = 2.2204e-16)
        self.lcc = CorrelationCoefficient()
        self.sim = Similarity()
        self.nss = NormalizedScanpathSaliency()
    
    def forward(self, pred, y):
        sal_maps, fix_maps = y
        
        _, _, h, w = fix_maps.shape
        
        large_pred = interpolate(pred, size = (h, w), mode = "bilinear")
        large_pred = self.batch_flatten(large_pred)
        
        pred = self.batch_flatten(pred)
        pred = self.normalize_to_probabilitic(pred)
        
        sal_maps = self.batch_flatten(sal_maps)
        sal_maps = self.normalize_to_probabilitic(sal_maps)
        
        fix_maps = self.batch_flatten(fix_maps)
        
        kld_loss = 10.0 * self.kld(pred, sal_maps)
        lcc_loss = -2.0 * self.lcc(pred, sal_maps)
        sim_loss = -1.0 * self.sim(pred, sal_maps)
        nss_loss = -1.0 * self.nss(large_pred, fix_maps)
        
        loss = kld_loss + lcc_loss + sim_loss + nss_loss
        
        return loss
```

## Training (CUDA only)

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
from torch.nn.functional import interpolate
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
        from measure.measures import *
        from measure.transforms import *
        
        self.batch_flatten = BatchFlatten()
        self.normalize_to_probabilitic = NormalizeToProbabilitic()
        
        self.kld = KLDivergence(eps = 2.2204e-16)
        self.lcc = CorrelationCoefficient()
        self.sim = Similarity()
        self.nss = NormalizedScanpathSaliency()
    
    def forward(self, pred, y):
        sal_maps, fix_maps = y
        
        _, _, h, w = fix_maps.shape
        
        large_pred = interpolate(pred, size = (h, w), mode = "bilinear")
        large_pred = self.batch_flatten(large_pred)
        
        pred = self.batch_flatten(pred)
        pred = self.normalize_to_probabilitic(pred)
        
        sal_maps = self.batch_flatten(sal_maps)
        sal_maps = self.normalize_to_probabilitic(sal_maps)
        
        fix_maps = self.batch_flatten(fix_maps)
        
        kld_loss = 10.0 * self.kld(pred, sal_maps)
        lcc_loss = -2.0 * self.lcc(pred, sal_maps)
        sim_loss = -1.0 * self.sim(pred, sal_maps)
        nss_loss = -1.0 * self.nss(large_pred, fix_maps)
        
        loss = kld_loss + lcc_loss + sim_loss + nss_loss
        
        return loss

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

train(wrapper, "transalnet", train_dataset, val_dataset, gpus = 2, batch_size = 4, max_epochs = 30)
```

## Models

The `models` module provides some saliency models, including `TranSalNet-Dense`, `MSINet`, `CASNet` and `SalGAN`.