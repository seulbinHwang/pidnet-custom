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
from utils import class_info
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
    def __init__(
            self,
            root,  # data/
            list_path,  # list/ade/training.odgt
            num_classes=150,  # 2
            multi_scale=True,  # True
            flip=True,  # True
            ignore_label=255,
            base_size=2048,
            crop_size=(512, 1024),  # (1024, 1024)
            scale_factor=16,  # 16
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

        self.root = root  # data/
        self.list_path = list_path  # list/ade/training.odgt
        self.num_classes = num_classes
        self.low_resolution = low_resolution
        self.multi_scale = multi_scale
        self.flip = flip
        # data/list/cityscapes/train.lst
        # 예시: leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        # gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        data_list_dir = os.path.join(root, list_path)
        # List[Dict[str, Any]]
        self.img_list = self.parse_input_list(data_list_dir)
        if self.num_classes == 3:
            self.target_classes = ["person", "grass"]  # "ball"
            if IS_MAC:
                # check
                self.class_weights = torch.FloatTensor([
                    1.0023,
                    1.5023,
                    0.1043,
                ]).to(device=device)
            else:
                self.class_weights = torch.FloatTensor([
                    1.0023,
                    1.5023,
                    0.1043,
                ]).cuda()
        elif self.num_classes == 4:
            self.target_classes = ["person", "grass", "ball"]
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
        assert len(self.target_classes) + 1 == self.num_classes
        UNKNOWN_CLASS = len(self.target_classes)
        self.class_info = class_info.ADE()
        self.class_to_idx = self.class_info.class_to_idx
        self.idx_to_class = self.class_info.idx_to_class
        self.label_mapping = {
            idx: UNKNOWN_CLASS
            for idx in range(256)
        }
        # self.label_mapping = {
        #     idx: UNKNOWN_CLASS
        #     for idx in range(len(self.idx_to_class))
        # } # length: 150
        for idx, target_class in enumerate(self.target_classes):
            self.label_mapping[self.class_to_idx[target_class]] = idx
        self.bd_dilate_size = bd_dilate_size # 4
        # List[Dict[str, str]] # img, label, name
        self.files = self.read_files()

    def read_files(self) -> List[Dict[str, str]]:
        files = []
        # list_path: "list/ade/training.odgt"
        # img_list: List[Dict[str, Any]]
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
            """
{
"fpath_img": "ADEChallengeData2016/images/training/ADE_train_00000001.jpg",
"fpath_segm": "ADEChallengeData2016/annotations/training/ADE_train_00000001.png",
"width": 683,
"height": 512
}
            """
            # img_list: List[Dict[str, Any]]
            for item in self.img_list:
                # item: Dict[str, Any]
                image_path = item["fpath_img"]
                label_path = item["fpath_segm"]
                width = item["width"]
                height = item["height"]
                name = os.path.splitext(
                    os.path.basename(label_path))[0]  # ADE_train_00000001
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "width": width,
                    "height": height,
                })
        return files

    def convert_label(self, label, inverse=False):
        # # Get the unique elements and their counts
        # unique_elements, counts = np.unique(label, return_counts=True)
        #
        # # Count the number of different values
        # num_different_values = len(unique_elements)
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                # v에 ignore label이 있음.
                label[temp == k] = v
        # Get the unique elements and their counts
        unique_elements, counts = np.unique(label, return_counts=True)
        # Count the number of different values
        num_different_values = len(unique_elements)
        assert num_different_values <= self.num_classes, \
            f"num_different_values: {num_different_values}, " \
            f"self.num_classes: {self.num_classes}"
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
        # self.files: List[Dict[str, str]]
        item = self.files[index]
        name = item["name"]  # aADE_train_00000001
        if self.low_resolution:
            # TODO: Check!
            # root: data/
            """
"img": "ADEChallengeData2016/images/training/ADE_train_00000001.jpg",
"label": "ADEChallengeData2016/annotations/training/ADE_train_00000001.png",
"name": "ADE_train_00000001",
"width": 683,
"height": 512,
            """
            # leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
            image_path = os.path.join(self.root, 'ade_resized', item["img"])
        else:
            image_path = os.path.join(self.root, 'ade', item["img"])
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        size = image.shape  # (H, w, 3)

        if 'test' in self.list_path:
            # TODO: check!
            image = self.input_transform(image)  # (H, w, 3)
            image = image.transpose((2, 0, 1))  # (3, H, w)
            return image.copy(), np.array(size), name
        # label: gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        if self.low_resolution:
            # TODO: Check!
            label_path = os.path.join(self.root, 'ade_resized', item["label"])

        else:
            label_path = os.path.join(self.root, 'ade', item["label"])
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED).astype(np.int16)
        label += -1
        label = label.astype(np.uint8)
        # (1024, 2048)
        label = self.convert_label(label)
        # Let width and height of image and label is divisible by 8.
        # print shape
        H, W, C = image.shape

        # H, W 가 8의 배수가 아니면, 8의 배수가 되도록.
        new_H = int(np.ceil(H / 8) * 8)
        new_W = int(np.ceil(W / 8) * 8)

        # 리사이즈
        image = cv2.resize(image, (new_W, new_H))
        label = cv2.resize(label, (new_W, new_H),
                           interpolation=cv2.INTER_NEAREST)
        image, label, edge = self.gen_sample(image,
                                             label,
                                             self.multi_scale,
                                             self.flip,
                                             edge_size=self.bd_dilate_size)
        # image: (3, H, W)
        # label: (H, W)
        # edge: (H, W)
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
