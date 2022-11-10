from torch.utils.data import Dataset
from .saliency_map_dataset import SaliencyMapDataset
from typing import List

__all__ = ["Subdataset"]

class Subdataset(Dataset):
    def __init__(self, upstream: Dataset, indices: List[int]):
        self.upstream = upstream
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.upstream[self.indices[index]]

    def to_saliency_map_dataset(self) -> SaliencyMapDataset:
        class SaliencyMapSubdataset(SaliencyMapDataset):
            def __init__(self, upstream: SaliencyMapDataset, indices: List[int]):
                super().__init__(upstream.image_transform, upstream.ground_truth_transform)
                self.upstream = upstream
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def get_image_path(self, index: int) -> str:
                return self.upstream.get_image_path(self.indices[index])

            def get_ground_truth_path(self, index: int) -> str:
                return self.upstream.get_ground_truth_path(self.indices[index])

        assert isinstance(self.upstream, SaliencyMapDataset)
        return SaliencyMapSubdataset(self.upstream, self.indices)