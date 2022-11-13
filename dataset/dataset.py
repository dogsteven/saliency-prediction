from torch.nn import Module
from torch.utils.data import Dataset
from .image_dataset import ImageDataset
from matplotlib import pyplot
from PIL import Image

__all__ = ["TransformedDataset", "ZippedDataset"]

class TransformedDataset(Dataset):
    def __init__(self, upstream: Dataset, transform):
        self.upstream = upstream
        self.transform = transform

    def __len__(self):
        return len(self.upstream)

    def __getitem__(self, index: int):
        return self.transform(self.upstream[index])

class ZippedDataset(Dataset):
    def __init__(self, first_dataset: Dataset, second_dataset: Dataset):
        self.first_dataset = first_dataset
        self.second_dataset = second_dataset

    def __len__(self):
        first_len = len(self.first_dataset)
        second_len = len(self.second_dataset)
        return min(first_len, second_len)

    def __getitem__(self, index: int):
        return self.first_dataset[index], self.second_dataset[index]

    def test(self, model: Module, start: int, end: int, gpu = False):
        if not isinstance(self.first_dataset, ImageDataset) or not isinstance(self.second_dataset, ImageDataset):
            return

        if gpu:
            model.cuda()
        else:
            model.cpu()

        _, axs = pyplot.subplots(nrows = end - start, ncols = 3, figsize = (30, 10 * (end - start)))

        for index in range(start, end):
            image = Image.open(self.first_dataset.get_path(index)).convert("RGB")
            ground_truth = Image.open(self.second_dataset.get_path(index)).convert("L")

            input_batch = self.first_dataset.transform(image).unsqueeze(0)

            if gpu:
                input_batch = input_batch.cuda()

            predicted = model(input_batch).squeeze(0)

            if gpu:
                predicted = predicted.cpu()

            predicted = predicted.detach().permute(1, 2, 0)

            axs[index - start][0].imshow(image)
            axs[index - start][1].imshow(ground_truth)
            axs[index - start][2].imshow(predicted)

