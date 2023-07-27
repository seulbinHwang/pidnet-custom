# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os
from typing import List, Dict, Tuple, Any
import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset
import platform

import torch

UNKNOWN_CLASS = 2
# UNKNOWN_CLASS = 1

def get_torch_gpu_device(gpu_idx: int = 0) -> str:
    if IS_MAC:
        assert torch.backends.mps.is_available()
        device = f"mps:{gpu_idx}"
    else:
        assert torch.cuda.is_available()
        device = f"cuda:{gpu_idx}"
    return device


if platform.system() == "Darwin" and platform.uname().processor == "arm":
    IS_MAC = True
    device = get_torch_gpu_device()
else:
    IS_MAC = False


class ADE(BaseDataset):

    def __init__(self,
                 root, # data/
                 list_path, # list/ade/training.odgt
                 num_classes=150,# 2
                 multi_scale=True, # True
                 flip=True,# True
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024), # (1024, 1024)
                 scale_factor=16, # 16
                 low_resolution=False,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(ADE, self).__init__(
            ignore_label,
            base_size,
            crop_size,
            scale_factor,
            mean,
            std,
        )

        self.root = root # data/
        # list/ade/training.odgt
        self.list_path = list_path
        self.num_classes = num_classes
        self.low_resolution = low_resolution
        self.multi_scale = multi_scale
        self.flip = flip
        # data/list/cityscapes/train.lst
        # 예시: leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        # gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        data_list_dir = os.path.join(root, list_path)
        if list_path.endswith('.lst'):
            self.img_list: List[List[str]] = [
                line.strip().split() for line in open(data_list_dir)
            ]
        else:
            self.img_list: List[Dict[str, Any]] = self.parse_input_list(data_list_dir)

        # List[Dict[str, str]] # img, label, name
        self.files: List[Dict[str, str]] = self.read_files()
        # check
        self.idx_to_class = {
  "0": "wall",
  "1": "building",
  "2": "sky",
  "3": "floor",
  "4": "tree",
  "5": "ceiling",
  "6": "road",
  "7": "bed ",
  "8": "windowpane",
  "9": "grass",
  "10": "cabinet",
  "11": "sidewalk",
  "12": "person",
  "13": "earth",
  "14": "door",
  "15": "table",
  "16": "mountain",
  "17": "plant",
  "18": "curtain",
  "19": "chair",
  "20": "car",
  "21": "water",
  "22": "painting",
  "23": "sofa",
  "24": "shelf",
  "25": "house",
  "26": "sea",
  "27": "mirror",
  "28": "rug",
  "29": "field",
  "30": "armchair",
  "31": "seat",
  "32": "fence",
  "33": "desk",
  "34": "rock",
  "35": "wardrobe",
  "36": "lamp",
  "37": "bathtub",
  "38": "railing",
  "39": "cushion",
  "40": "base",
  "41": "box",
  "42": "column",
  "43": "signboard",
  "44": "chest of drawers",
  "45": "counter",
  "46": "sand",
  "47": "sink",
  "48": "skyscraper",
  "49": "fireplace",
  "50": "refrigerator",
  "51": "grandstand",
  "52": "path",
  "53": "stairs",
  "54": "runway",
  "55": "case",
  "56": "pool table",
  "57": "pillow",
  "58": "screen door",
  "59": "stairway",
  "60": "river",
  "61": "bridge",
  "62": "bookcase",
  "63": "blind",
  "64": "coffee table",
  "65": "toilet",
  "66": "flower",
  "67": "book",
  "68": "hill",
  "69": "bench",
  "70": "countertop",
  "71": "stove",
  "72": "palm",
  "73": "kitchen island",
  "74": "computer",
  "75": "swivel chair",
  "76": "boat",
  "77": "bar",
  "78": "arcade machine",
  "79": "hovel",
  "80": "bus",
  "81": "towel",
  "82": "light",
  "83": "truck",
  "84": "tower",
  "85": "chandelier",
  "86": "awning",
  "87": "streetlight",
  "88": "booth",
  "89": "television receiver",
  "90": "airplane",
  "91": "dirt track",
  "92": "apparel",
  "93": "pole",
  "94": "land",
  "95": "bannister",
  "96": "escalator",
  "97": "ottoman",
  "98": "bottle",
  "99": "buffet",
  "100": "poster",
  "101": "stage",
  "102": "van",
  "103": "ship",
  "104": "fountain",
  "105": "conveyer belt",
  "106": "canopy",
  "107": "washer",
  "108": "plaything",
  "109": "swimming pool",
  "110": "stool",
  "111": "barrel",
  "112": "basket",
  "113": "waterfall",
  "114": "tent",
  "115": "bag",
  "116": "minibike",
  "117": "cradle",
  "118": "oven",
  "119": "ball",
  "120": "food",
  "121": "step",
  "122": "tank",
  "123": "trade name",
  "124": "microwave",
  "125": "pot",
  "126": "animal",
  "127": "bicycle",
  "128": "lake",
  "129": "dishwasher",
  "130": "screen",
  "131": "blanket",
  "132": "sculpture",
  "133": "hood",
  "134": "sconce",
  "135": "vase",
  "136": "traffic light",
  "137": "tray",
  "138": "ashcan",
  "139": "fan",
  "140": "pier",
  "141": "crt screen",
  "142": "plate",
  "143": "monitor",
  "144": "bulletin board",
  "145": "shower",
  "146": "radiator",
  "147": "glass",
  "148": "clock",
  "149": "flag"
}
        self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        self.target_classes = ["person", "grass", "ball"]
        assert len(self.target_classes) + 1 == self.num_classes
        UNKNOWN_CLASS = len(self.idx_to_class)
        self.label_mapping = {idx: UNKNOWN_CLASS for idx in range(len(self.idx_to_class))}
        for idx, target_class in enumerate(self.target_classes):
            self.label_mapping[self.class_to_idx[target_class]] = idx
        # if IS_MAC:
        #     # check
        #     self.class_weights = torch.FloatTensor([
        #         1.0023,
        #         0.0843,
        #     ]).to(device=device)
        # else:
        #     self.class_weights = torch.FloatTensor([
        #         1.0023,
        #         0.0843,
        #     ]).cuda()
        if self.num_classes == 4:
            if IS_MAC:
                # check
                self.class_weights = torch.FloatTensor([
                    1.0023,
                    1.0023,
                    2.0023,
                    0.1043,
                ]).to(device=device)
            else:
                self.class_weights = torch.FloatTensor([
                    1.0023,
                    1.0023,
                    2.0023,
                    0.1043,
                ]).cuda()
        else:
            raise NotImplementedError
        self.bd_dilate_size = bd_dilate_size

    def read_files(self) -> List[Dict[str, str]]:
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item["fpath_img"]
                width = item["width"]
                height = item["height"]
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                    "width": width,
                    "height": height,
                })
        else:
            # image_path:
            # leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
            # label_path:
            # gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
            # name:
            # aachen_000000_000019_gtFine_labelIds
            """
{
"fpath_img": "ADEChallengeData2016/images/training/ADE_train_00000001.jpg",
"fpath_segm": "ADEChallengeData2016/annotations/training/ADE_train_00000001.png",
"width": 683,
"height": 512
}
            """
            for item in self.img_list:
                image_path = item["fpath_img"]
                label_path = item["fpath_segm"]
                width = item["width"]
                height = item["height"]
                name = os.path.splitext(os.path.basename(label_path))[0] # ADE_train_00000001
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "width": width,
                    "height": height,
                })
        return files

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                # v에 ignore label이 있음.
                label[temp == k] = v
        return label

    def __getitem__(
            self, index
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
        """
        Args:
            index:

        Returns:
            image: (3, height, width)
            label: (height, width)
                0, 1, 255 로만 이루어져 있음.
            edge: (height, width)
                0, 1(edge임) 로만 이루어져 있음
        """
        item = self.files[index]
        name = item["name"]  # aachen_000000_000019_gtFine_labelIds
        if self.low_resolution:
            # root: data/
            """
"img": "ADEChallengeData2016/images/training/ADE_train_00000001.jpg",
"label": "ADEChallengeData2016/annotations/training/ADE_train_00000001.png",
"name": "ADE_train_00000001",
"width": 683,
"height": 512,
            """
            # leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
            path = os.path.join(self.root, 'ade_resized', item["img"])
        else:
            path = os.path.join(self.root, 'ade', item["img"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        size = image.shape  # (H, w, 3)

        if 'test' in self.list_path:
            # TODO: check!
            image = self.input_transform(image)  # (H, w, 3)
            image = image.transpose((2, 0, 1))  # (3, H, w)
            return image.copy(), np.array(size), name
        # label: gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        if self.low_resolution:
            path = os.path.join(self.root, 'ade_resized', item["label"])

        else:
            path = os.path.join(self.root, 'ade', item["label"])
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        label += -1
        # (1024, 2048)
        label = self.convert_label(label)
        # Let width and height of image and label is divisible by 8.
        image = image[:-(image.shape[0] % 8), :-(image.shape[1] % 8), :]
        label = label[:-(label.shape[0] % 8), :-(label.shape[1] % 8)]
        image, label, edge = self.gen_sample(image,
                                             label,
                                             self.multi_scale,
                                             self.flip,
                                             edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
