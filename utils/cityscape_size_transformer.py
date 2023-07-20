import os
import random
from typing import Tuple, List, Dict
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
                    # new_filename = f"{os.path.splitext(filename)[0]}_{new_width}x{new_height}{os.path.splitext(filename)[1]}"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    resized_img.save(os.path.join(save_path, filename))


def read_files(img_list) -> List[Dict[str, str]]:
    files = []
    if 'test' in list_path:
        for item in img_list:
            image_path = item
            name = os.path.splitext(os.path.basename(image_path[0]))[0]
            files.append({
                "img": image_path[0],
                "name": name,
            })
    else:
        # image_path:
        # leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        # label_path:
        # gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        # name:
        # aachen_000000_000019_gtFine_labelIds
        for item in img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({
                "img": image_path,
                "label": label_path,
                "name": name
            })
    return files



"""
1. get pngs list from “PycharmProjects/pidnet-custom/data/list/cityscapes/train.lst”
    1. [leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png, gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png …]
    2. make new “PycharmProjects/pidnet-custom/data/list/cityscapes_resized/train.lst”
    3. [leftImg8bit_resized/train/aachen/aachen_000000_000019_leftImg8bit.png, gtFine_resized/train/aachen/aachen_000000_000019_gtFine_labelIds.png …]
2. for pngs list
    1. 1024 * 2048 → below 240 * 320 
    2. save to  
        1. PycharmProjects/pidnet-custom/data/cityscapes/leftImg8bit_resized
        2. PycharmProjects/pidnet-custom/data/cityscapes/gtFine_resized
"""
root = "./data/"
list_path = "list/cityscapes/train.lst"
img_name_list = [
    line.strip().split() for line in open(root + list_path)
]
# img_name_list = [
#     'leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png',
#     'gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png'
#     # ...
# ]

# resized_img_name_list = []
#
# for item in img_name_list:
#     image_path, label_path = item
#
#     image_path = image_path.split("/")
#     image_path[0] += "_resized"
#     resize_image_path = "/".join(image_path)
#     label_path = label_path.split("/")
#     label_path[0] += "_resized"
#     resize_label_path = "/".join(label_path)
#     # Reconstruct the modified item
#     # Add the modified item to the 'resized_img_name_list' list
#     resized_img_name_list.append([resize_image_path, resize_label_path])
# file_path = "./data/list/cityscapes_resized/train.lst"
# if not os.path.exists(os.path.dirname(file_path)):
#     os.makedirs(os.path.dirname(file_path))
# with open(file_path, 'w') as file:
#     for resized_item in resized_img_name_list:
#         file.write(resized_item[0] + " " + resized_item[1] + "\n")

files: List[Dict[str, str]] = read_files(img_name_list)

kinds = ["train", "val" , "test"]
for kind in kinds:
    folder_path = f"./data/cityscapes/leftImg8bit/{kind}"
    save_path = f"./data/cityscapes_resized/leftImg8bit/{kind}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    min_size = (180, 180)
    new_size = ((320, 320), (240, 240))


    # print("Modified paths saved to:", file_path)
    # all folders in 'folder_path' will be resized and saved to 'save_path'
    all_folders = os.listdir(folder_path)
    for folder_name in all_folders:
        sub_save_path = os.path.join(save_path, folder_name)
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        sub_folder_path = os.path.join(folder_path, folder_name)
        resize_images_in_folder_randomly(sub_folder_path, sub_save_path, min_size, new_size)


    folder_path = f"./data/cityscapes/gtFine/{kind}"
    save_path = f"./data/cityscapes_resized/gtFine/{kind}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    min_size = (180, 180)
    new_size = ((320, 320), (240, 240))

    all_folders = os.listdir(folder_path)
    for folder_name in all_folders:
        sub_save_path = os.path.join(save_path, folder_name)
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        sub_folder_path = os.path.join(folder_path, folder_name)
        resize_images_in_folder_randomly(sub_folder_path, sub_save_path, min_size, new_size)