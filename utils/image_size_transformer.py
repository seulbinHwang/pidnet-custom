import os
import random
from PIL import Image
from typing import Tuple


def resize_images_in_folder_randomly(
        folder_path: str, save_path: str, min_size: Tuple[int, int],
        size_range: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
    """
    Resize images in the given folder to a random size.

    Args:
    folder_path (str): Path to the folder with images.
    min_size (tuple): Minimum width and height of the images to resize.
    size_range (tuple): Range of sizes for the resized images.
    """
    min_width, min_height = min_size
    width_range, height_range = size_range

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            file_path = os.path.join(folder_path, filename)
            with Image.open(file_path) as img:
                width, height = img.size
                if width >= min_width and height >= min_height:
                    new_width = random.randint(*width_range)
                    # new_width should be multiple of 8.
                    new_width = new_width - new_width % 8
                    new_height = random.randint(*height_range)
                    # new_height should be multiple of 8.
                    new_height = new_height - new_height % 8
                    resized_img = img.resize((new_width, new_height))
                    new_filename = f"{os.path.splitext(filename)[0]}_{new_width}x{new_height}{os.path.splitext(filename)[1]}"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    resized_img.save(os.path.join(save_path, new_filename))


folder_path = "./samples/raw_data_from_video"
save_path = "./samples/resized_data_from_video"
min_size = (512, 512)
new_size = ((2048, 2048), (1024, 1024))
resize_images_in_folder_randomly(folder_path, save_path, min_size, new_size)
