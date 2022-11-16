from torch import no_grad
from torch.utils.data import Dataset
from PIL import Image
from os.path import join
from os import mkdir
from .utilities import get_all_images_from_directory, parse_image_path

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

    def generate(self, model, path, directory, transform):
        from tqdm import tqdm

        output_path = join(path, directory)
        try:
            mkdir(output_path)
        except:
            print(f"The directory {directory} does exists in {path}")

        model.cuda()

        with no_grad():
            for index in tqdm(range(len(self))):
                image = self[index].cuda()
                predicted = model(image.unsqueeze(0)).squeeze(0).detach().cpu()
                output_image = transform(predicted)
                name, _ = parse_image_path(self.get_subpath(index))
                output_image.save(join(output_path, f"{name}.png"), "PNG")


class ImageDirectorySourceDataset(ImageDirectoryDataset):
    def __init__(self, directory, transform, mode):
        super().__init__(directory, transform, mode)
        self.subpaths = sorted(get_all_images_from_directory(self.directory))

    def __len__(self):
        return len(self.subpaths)

    def get_subpath(self, index: int) -> str:
        return self.subpaths[index]