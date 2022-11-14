from os import listdir
from typing import List, Tuple

__all__ = ["is_image_path", "get_all_images_from_directory", "parse_image_path"]

def is_image_path(path: str) -> bool:
    return path.endswith(".png") or path.endswith(".jpg") or path.endswith(".jpeg")

def get_all_images_from_directory(directory: str) -> List[str]:
    return [path for path in listdir(directory) if is_image_path(path)]

def parse_image_path(path: str) -> Tuple[str, str]:
    assert is_image_path(path)
    for ext in [".png", ".jpg", ".jpeg"]:
        if path.endswith(ext):
            return path[:-len(ext)], ext[1:]