from torch.utils.data import Dataset
from PIL import Image
from os.path import join
from .utilities import get_all_images_from_directory

__all__ = ["ImageDataset", "ImageDirectoryDataset", "ImageDirectorySourceDataset"]

class ImageDataset(Dataset):
    def __init__(self, transform, mode):
        self.transform = transform
        self.mode = mode

    def get_path(self, index: int) -> str:
        pass

    def __getitem__(self, index: int):
        return self.transform(Image.open(self.get_path(index)).convert(self.mode))

class ImageDirectoryDataset(ImageDataset):
    def __init__(self, directory, transform, mode):
        self.directory = directory
        super().__init__(transform, mode)

    def get_subpath(self, index: int) -> str:
        pass

    def get_path(self, index: int) -> str:
        return join(self.directory, self.get_subpath(index))

class ImageDirectorySourceDataset(ImageDirectoryDataset):
    def __init__(self, directory, transform, mode):
        super().__init__(directory, transform, mode)
        self.subpaths = sorted(get_all_images_from_directory(self.directory))

    def __len__(self):
        return len(self.subpaths)

    def get_subpath(self, index: int) -> str:
        return self.subpaths[index]