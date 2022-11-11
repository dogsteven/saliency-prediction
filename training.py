from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

__all__ = ["train"]

def train(model: LightningModule, name: str, train_dataset: Dataset, val_dataset: Dataset, gpus: int = 2, batch_size: int = 32, max_epochs: int = 100):
    if gpus < 1:
        return

    callbacks = [
        EarlyStopping(monitor = "val_loss"),
        ModelCheckpoint(dirpath = "./", filename = f"{name}-checkpoint.model", monitor = "val_loss")
    ]
    if gpus == 1:
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, pin_memory = True, shuffle = True)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size, pin_memory = True, shuffle = True)
        trainer = Trainer(
            accelerator = "gpu",
            devices = 1,
            callbacks = callbacks,
            max_epochs = max_epochs
        )

        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size = batch_size, pin_memory = True, shuffle = True, num_workers = gpus)
        val_dataloader = DataLoader(val_dataset, batch_size = batch_size, pin_memory = True, shuffle = False, num_workers = gpus)
        trainer = Trainer(
            accelerator = "gpu",
            devices = gpus,
            strategy = "ddp_fork",
            callbacks = callbacks,
            max_epochs = max_epochs
        )

        trainer.fit(model, train_dataloader, val_dataloader)