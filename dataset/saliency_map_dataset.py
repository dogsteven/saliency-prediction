import os
from PIL import Image
from torch import unsqueeze
from torch.utils.data import Dataset
from torch.nn import Module
from matplotlib import pyplot

__all__ = ["SaliencyMapDataset", "SaliencyMapDirectoryDataset"]

class SaliencyMapDataset(Dataset):
    def __init__(self, image_transform, ground_truth_transform):
        self.image_transform = image_transform
        self.ground_truth_transform = ground_truth_transform

    def get_image_path(self, index: int) -> str:
        pass

    def get_ground_truth_path(self, index: int) -> str:
        pass

    def __getitem__(self, index: int):
        image_path = self.get_image_path(index)
        ground_truth_path = self.get_ground_truth_path(index)

        image = self.image_transform(Image.open(image_path).convert("RGB"))
        ground_truth = self.ground_truth_transform(Image.open(ground_truth_path).convert("L"))

        return image, ground_truth

    def test(self, model: Module, start: int, end: int, cuda: bool = False):
        if cuda:
            model.cuda()

        _, axs = pyplot.subplots(nrows = end - start, ncols = 3, figsize = (30, 10 * (end - start)))

        for index in range(start, end):
            image = Image.open(self.get_image_path(index)).convert("RGB")
            ground_truth = Image.open(self.get_ground_truth_path(index)).convert("L")
            tensor_image = unsqueeze(self.image_transform(image), 0)

            if cuda:
                tensor_image = tensor_image.cuda()

            predicted = model(tensor_image)[0]

            if cuda:
                predicted = predicted.cpu()

            predicted = predicted.detach().permute(1, 2, 0)

            axs[index - start][0].imshow(image)
            axs[index - start][1].imshow(ground_truth)
            axs[index - start][2].imshow(predicted)

class SaliencyMapDirectoryDataset(SaliencyMapDataset):
    def __init__(self, image_directory, ground_truth_directory, image_transform, ground_truth_transform):
        super().__init__(image_transform, ground_truth_transform)
        self.image_directory = image_directory
        self.ground_truth_directory = ground_truth_directory

    def get_image_subpath(self, index: int) -> str:
        pass

    def get_ground_truth_subpath(self, index: int) -> str:
        pass

    def get_image_path(self, index: int) -> str:
        return os.path.join(self.image_directory, self.get_image_subpath(index))

    def get_ground_truth_path(self, index: int) -> str:
        return os.path.join(self.ground_truth_directory, self.get_ground_truth_subpath(index))