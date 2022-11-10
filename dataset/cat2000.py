from .saliency_map_dataset import SaliencyMapDirectoryDataset
from .subdataset import Subdataset
from typing import Tuple

__all__ = ["CAT2000SaliencyMapDataset"]

class CAT2000SaliencyMapDataset(SaliencyMapDirectoryDataset):
    categories = ["Action", "Affective", "Art", "BlackWhite", "Cartoon", "Fractal", "Indoor", "Inverted", "Jumbled", "LineDrawing", "LowResolution", "Noisy", "Object", "OutdoorManMade", "OutdoorNatural", "Pattern", "Random", "Satelite", "Sketch", "Social"]

    def __len__(self):
        return 2000

    def get_image_subpath(self, index: int) -> str:
        category = CAT2000SaliencyMapDataset.categories[index % 20]
        item_name = f"{str(2 * (index // 20) + 1).zfill(3)}.jpg"
        return f"{category}/{item_name}"

    def get_ground_truth_subpath(self, index: int) -> str:
        category = CAT2000SaliencyMapDataset.categories[index % 20]
        item_name = f"{str(2 * (index // 20) + 1).zfill(3)}.jpg"
        return f"{category}/{item_name}"

    def split(self, train_ratio: float = 0.8) -> Tuple[Subdataset, Subdataset]:
        import random

        train_indices = []
        val_indices = []

        for i in range(20):
            x = list(range(100))
            y = []

            for _ in range(int((1.0 - train_ratio) * 100)):
                index = random.choice(x)
                y.append(index)
                x.remove(index)

            train_indices = train_indices + [index * 20 + i for index in x]
            val_indices = val_indices + [index * 20 + i for index in y]

        return Subdataset(self, train_indices), Subdataset(self, val_indices)