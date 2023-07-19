import os
import cv2
import numpy as np

ignore_label = 255
label_mapping = {
    -1: ignore_label,  #
    0: ignore_label,
    1: ignore_label,
    2: ignore_label,
    3: ignore_label,
    4: ignore_label,
    5: ignore_label,
    6: ignore_label,
    7: 0,  # road
    8: 1,  # sidewalk
    9: ignore_label,
    10: ignore_label,
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: ignore_label,
    15: ignore_label,
    16: ignore_label,
    17: 5,  # pole
    18: ignore_label,
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: ignore_label,
    30: ignore_label,
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18  # bicycle /
}


def convert_label(label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


root = "data/"
list_path = "list/cityscapes/train.lst"
img_list = [line.strip().split() for line in open(root + list_path)]
files = []
for item in img_list:
    image_path, label_path = item
    name = os.path.splitext(os.path.basename(label_path))[0]
    files.append({"img": image_path, "label": label_path, "name": name})
#### __getitem__ ####
index = 0
item = files[index]
name = item["name"]
path = os.path.join(root, 'cityscapes', item["img"])
print("path: ", path)
image = cv2.imread(path, cv2.IMREAD_COLOR)
size = image.shape

label = cv2.imread(os.path.join(root, 'cityscapes', item["label"]),
                   cv2.IMREAD_GRAYSCALE)
print("label shape: ", label.shape)
print("label: ", label)
label = convert_label(label)
print("------------------")
print("label shape: ", label.shape)
print("label: ", label)
