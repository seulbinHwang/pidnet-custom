# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os
from typing import List, Dict, Tuple
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
                 root,
                 list_path,
                 num_classes=19,
                 multi_scale=True,
                 flip=True,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024),
                 scale_factor=16,
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

        self.root = root # 'data/'
        # 'list/ade/train.lst'
        self.list_path = list_path
        self.num_classes = num_classes
        self.low_resolution = low_resolution
        self.multi_scale = multi_scale
        self.flip = flip
        # data/list/ade/train.lst
        # e.g.
        # leftImg8bit/train/unclassified/outliers__bookshelf/ADE_train_00000936_leftImg8bit.jpg
        # gtFine/train/unclassified/outliers__bookshelf/ADE_train_00000936_gtFine_labelIds.jpg
        # 예시:
        # leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        # gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        self.img_list = [
            line.strip().split() for line in open(root + list_path)
        ]
        # List[Dict[str, str]] # img, label, name
        self.files: List[Dict[str, str]] = self.read_files()
        # check
        self.label_mapping = {
            -1: UNKNOWN_CLASS,  #
            0: UNKNOWN_CLASS,
            1: UNKNOWN_CLASS,
            2: UNKNOWN_CLASS,
            3: UNKNOWN_CLASS,
            4: UNKNOWN_CLASS,
            5: UNKNOWN_CLASS,
            6: UNKNOWN_CLASS,
            7: UNKNOWN_CLASS,  # road
            8: UNKNOWN_CLASS,  # sidewalk
            9: UNKNOWN_CLASS,
            10: UNKNOWN_CLASS,
            11: UNKNOWN_CLASS,  # building
            12: UNKNOWN_CLASS,  # wall
            13: UNKNOWN_CLASS,  # fence
            14: UNKNOWN_CLASS,
            15: UNKNOWN_CLASS,
            16: UNKNOWN_CLASS,
            17: UNKNOWN_CLASS,  # pole
            18: UNKNOWN_CLASS,
            19: UNKNOWN_CLASS,  # traffic light
            20: UNKNOWN_CLASS,  # traffic sign
            21: UNKNOWN_CLASS,  # vegetation
            22: 1,  # terrain
            23: UNKNOWN_CLASS,  # sky
            24: 0,  # person
            25: 0,  # rider
            26: UNKNOWN_CLASS,  # car
            27: UNKNOWN_CLASS,  # truck
            28: UNKNOWN_CLASS,  # bus
            29: UNKNOWN_CLASS,
            30: UNKNOWN_CLASS,
            31: UNKNOWN_CLASS,  # train
            32: UNKNOWN_CLASS,  # motorcycle
            33: UNKNOWN_CLASS  # bicycle /
        }
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
        if IS_MAC:
            # check
            self.class_weights = torch.FloatTensor([
                1.0023,
                0.3000,
                0.0843,
            ]).to(device=device)
        else:
            self.class_weights = torch.FloatTensor([
                1.0023,
                0.3000,
                0.0843,
            ]).cuda()

        self.bd_dilate_size = bd_dilate_size

    def read_files(self) -> List[Dict[str, str]]:
        files = []
        # 'list/ade/train.lst'
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            # leftImg8bit/train/unclassified/outliers__bookshelf/ADE_train_00000936_leftImg8bit.jpg
            # gtFine/train/unclassified/outliers__bookshelf/ADE_train_00000936_gtFine_labelIds.jpg
            # image_path:
            # leftImg8bit/train/unclassified/outliers__bookshelf/ADE_train_00000936_leftImg8bit.jpg
            # label_path:
            # gtFine/train/unclassified/outliers__bookshelf/ADE_train_00000936_gtFine_labelIds.jpg
            # name:
            # ADE_train_00000936_gtFine_labelIds
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
###### 07.26
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
            # leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
            path = os.path.join(self.root, 'cityscapes_resized', item["img"])
        else:
            path = os.path.join(self.root, 'cityscapes', item["img"])
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        size = image.shape  # (H, w, 3)

        if 'test' in self.list_path:
            image = self.input_transform(image)  # (H, w, 3)
            image = image.transpose((2, 0, 1))  # (3, H, w)
            return image.copy(), np.array(size), name
        # label: gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        if self.low_resolution:
            path = os.path.join(self.root, 'cityscapes_resized', item["label"])

        else:
            path = os.path.join(self.root, 'cityscapes', item["label"])
        label = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # (1024, 2048)
        label = self.convert_label(label)
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
