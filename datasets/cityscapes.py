# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import os
from typing import List, Dict
import cv2
import numpy as np
from PIL import Image

import torch
from .base_dataset import BaseDataset

class Cityscapes(BaseDataset):
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
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                 bd_dilate_size=4):

        super(Cityscapes, self).__init__(ignore_label, base_size,
                crop_size, scale_factor, mean, std,)

        self.root = root
        # 'list/cityscapes/train.lst'
        self.list_path = list_path
        self.num_classes = num_classes

        self.multi_scale = multi_scale
        self.flip = flip
        # data/list/cityscapes/train.lst
        # 예시: leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
        # gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        self.img_list = [line.strip().split() for line in open(root+list_path)]
        # List[Dict[str, str]] # img, label, name
        self.files = self.read_files()

        self.label_mapping = {
            -1: ignore_label,  #
            0: ignore_label,
            1: ignore_label,
            2: ignore_label,
            3: ignore_label,
            4: ignore_label,
            5: ignore_label,
            6: ignore_label,
            7: ignore_label,  # road
            8: ignore_label,  # sidewalk
            9: ignore_label,
            10: ignore_label,
            11: ignore_label,  # building
            12: ignore_label,  # wall
            13: ignore_label,  # fence
            14: ignore_label,
            15: ignore_label,
            16: ignore_label,
            17: ignore_label,  # pole
            18: ignore_label,
            19: ignore_label,  # traffic light
            20: ignore_label,  # traffic sign
            21: ignore_label,  # vegetation
            22: 0,  # terrain
            23: ignore_label,  # sky
            24: 1,  # person
            25: ignore_label,  # rider
            26: ignore_label,  # car
            27: ignore_label,  # truck
            28: ignore_label,  # bus
            29: ignore_label,
            30: ignore_label,
            31: ignore_label,  # train
            32: ignore_label,  # motorcycle
            33: ignore_label  # bicycle /
        }
        self.class_weights = torch.FloatTensor([ 1.0023, 0.9843,]).cuda()
        
        self.bd_dilate_size = bd_dilate_size
    
    def read_files(self) -> List[Dict[str, str]]:
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
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
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"] # aachen_000000_000019_gtFine_labelIds
        image = cv2.imread(os.path.join(self.root,'cityscapes',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name
        # label: gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
        label = cv2.imread(os.path.join(self.root,'cityscapes',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        # (1024, 2048)
        label = self.convert_label(label)

        image, label, edge = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, edge_size=self.bd_dilate_size)

        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    
    def single_scale_inference(self, config, model, image):
        pred = self.inference(config, model, image)
        return pred


    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
