from .dataset import ZippedDataset
from .image_dataset import ImageDirectorySourceDataset

__all__ = ["SALICONSaliencyDataset"]

def SALICONSaliencyDataset(image_directory, image_transform, saliency_map_directory = None, saliency_map_transform = None, fixation_map_directory = None, fixation_map_transform = None):
    assert not (saliency_map_directory is None and fixation_map_directory is None)
    assert (saliency_map_directory is None and saliency_map_transform is None) or (saliency_map_directory is not None and saliency_map_transform is not None)
    assert (fixation_map_directory is None and fixation_map_transform is None) or (fixation_map_directory is not None and fixation_map_transform is not None)

    image_dataset = ImageDirectorySourceDataset(image_directory, image_transform, "RGB")

    if saliency_map_directory is not None and fixation_map_directory is not None:
        saliency_map_dataset = ImageDirectorySourceDataset(saliency_map_directory, saliency_map_transform, "L")
        fixation_map_dataset = ImageDirectorySourceDataset(fixation_map_directory, fixation_map_transform, "L")

        return ZippedDataset(
            image_dataset,
            ZippedDataset(
                saliency_map_dataset,
                fixation_map_dataset
            )
        )
    elif saliency_map_directory is not None:
        saliency_map_dataset = ImageDirectorySourceDataset(saliency_map_directory, saliency_map_transform, "L")

        return ZippedDataset(
            image_dataset,
            saliency_map_dataset
        )
    else:
        fixation_map_dataset = ImageDirectorySourceDataset(fixation_map_directory, fixation_map_transform, "L")

        return ZippedDataset(
            image_dataset,
            fixation_map_dataset
        )