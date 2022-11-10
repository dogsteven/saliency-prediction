import os
from .saliency_map_dataset import SaliencyMapDirectoryDataset

__all__ = ["SALICONSaliencyMapDataset"]

def is_image_path(path: str) -> bool:
    return path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg")

class SALICONSaliencyMapDataset(SaliencyMapDirectoryDataset):
    def __init__(self, image_directory, ground_truth_directory, image_transform, ground_truth_transform):
        super().__init__(image_directory, ground_truth_directory, image_transform, ground_truth_transform)

        self.image_subpaths = sorted([path for path in os.listdir(self.image_directory) if is_image_path(path)])
        self.ground_truth_subpaths = sorted([path for path in os.listdir(self.ground_truth_directory) if is_image_path(path)])

    def __len__(self):
        return len(self.image_subpaths)

    def get_image_subpath(self, index: int) -> str:
        return self.image_subpaths[index]

    def get_ground_truth_subpath(self, index: int) -> str:
        return self.ground_truth_subpaths[index]