from os import listdir
from typing import List

__all__ = ["is_image_path", "get_all_images_from_directory"]

def is_image_path(path: str) -> bool:
    return path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg")

def get_all_images_from_directory(directory: str) -> List[str]:
    return [path for path in listdir(directory) if is_image_path(path)]