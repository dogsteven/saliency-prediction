from torch.utils.data import Dataset
from PIL import Image
import os

__all__ = ["ImageDataset", "ImageDirectoryDataset"]

class ImageDataset(Dataset):
    def __init__(self, transform, mode: str = "RGB"):
        self.transform = transform
        self.mode = mode

    def get_image_path(self, index: int) -> str:
        pass

    def __getitem__(self, index: int):
        image_path = self.get_image_path(index)
        image = self.transform(Image.open(image_path).convert(self.mode))
        return image

class ImageDirectoryDataset(ImageDataset):
    def __init__(self, directory, transform, mode: str = "RGB"):
        super().__init__(transform, mode)
        self.directory = directory

    def get_image_subpath(self, index: int) -> str:
        pass

    def get_image_path(self, index: int) -> str:
        return os.path.join(self.directory, self.get_image_subpath(index))