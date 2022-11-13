from .dataset import ZippedDataset
from .image_dataset import ImageDirectoryDataset

__all__ = ["CAT2000SaliencyDataset"]

def CAT2000SaliencyDataset(image_directory, image_transform, ground_truth_directory, ground_truth_transform):
    class CAT2000ImageDataset(ImageDirectoryDataset):
        categories = ["Action", "Affective", "Art", "BlackWhite", "Cartoon", "Fractal", "Indoor", "Inverted", "Jumbled", "LineDrawing", "LowResolution", "Noisy", "Object", "OutdoorManMade", "OutdoorNatural", "Pattern", "Random", "Satelite", "Sketch", "Social"]

        def __len__(self):
            return 2000

        def get_subpath(self, index: int) -> str:
            category = CAT2000ImageDataset.categories[index % 20]
            item_name = f"{str(2 * (index // 20) + 1).zfill(3)}.jpg"
            return f"{category}/{item_name}"

    image_dataset = CAT2000ImageDataset(image_directory, image_transform, "RGB")
    ground_truth_dataset = CAT2000ImageDataset(ground_truth_directory, ground_truth_transform, "L")

    return ZippedDataset(image_dataset, ground_truth_dataset)