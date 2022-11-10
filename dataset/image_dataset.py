from torch.utils.data import Dataset
from PIL import Image
import os
from typing import Tuple

__all__ = ["ImageDataset", "ImageDirectoryDataset"]

def parse(path: str) -> Tuple[str, str]:
    exts = [".png", ".jpg", ".jpeg"]
    for ext in exts:
        if path.endswith(ext):
            return path[0:-len(ext)], ext[1:]

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

    def generate(self, model, path: str, name: str, transform):
        directory_path = os.path.join(path, name)
        try:
            os.mkdir(directory_path)
        except:
            print(f"Directory {name} already exsits")

        for index in range(len(self)):
            image = self[index].cuda().unsqueeze(0)
            predicted = model(image).squeeze(0)
            predicted = predicted.cpu().detach()
            predicted = transform(predicted)

            name, _ = parse(self.get_image_subpath(index))
            name = f"{name}.png"
            path = os.path.join(directory_path, name)

            predicted.save(path, "PNG")