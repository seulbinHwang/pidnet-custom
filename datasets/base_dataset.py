# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------
import json
from typing import List, Dict, Any
import cv2
import numpy as np
import random

from torch.nn import functional as F
from torch.utils import data

y_k_size = 6
x_k_size = 6


class BaseDataset(data.Dataset):
    def __init__(self,
                 ignore_label=255,
                 base_size=2048,
                 crop_size=(512, 1024),
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size  # 2048
        self.crop_size = crop_size  # (1024, 1024)
        self.ignore_label = ignore_label  # 255

        self.mean = mean  # [0.485, 0.456, 0.406]
        self.std = std  # [0.229, 0.224, 0.225]
        self.scale_factor = scale_factor  # 16

        self.files = []

    def __len__(self):
        return len(self.files)

    def input_transform(self, image, city=True):
        if city:  # BGR로 주어짐
            image = image.astype(np.float32)[:, :, ::-1]
        else:  # RGB로 주어짐
            image = image.astype(np.float32)
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    def parse_input_list(self,
                         data_list_dir,
                         max_sample=-1,
                         start_idx=-1,
                         end_idx=-1) -> List[Dict[str, Any]]:
        """
        :param data_list_dir: "./data/training.odgt"
{
"fpath_img": "ADEChallengeData2016/images/training/ADE_train_00000001.jpg",
"fpath_segm": "ADEChallengeData2016/annotations/training/ADE_train_00000001.png",
"width": 683,
"height": 512
}
        :param max_sample:
        :param start_idx:
        :param end_idx:
        :return:

        """
        if isinstance(data_list_dir, list):
            list_sample = data_list_dir
        elif isinstance(data_list_dir, str):
            list_sample = [
                json.loads(x.rstrip()) for x in open(data_list_dir, 'r')
            ]
        else:
            raise Exception('Invalid data list dir.')
        if max_sample > 0:
            list_sample = list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:  # divide file list
            list_sample = list_sample[start_idx:end_idx]

        self.num_sample = len(list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
        return list_sample

    def label_transform(self, label):
        return np.array(label).astype(np.uint8)

    def pad_image(self, image, h, w, size, padvalue):
        pad_image = image.copy()
        pad_h = max(size[0] - h, 0)
        pad_w = max(size[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(src=image,
                                           top=0,
                                           bottom=pad_h,
                                           left=0,
                                           right=pad_w,
                                           borderType=cv2.BORDER_CONSTANT,
                                           value=padvalue)

        return pad_image

    def rand_crop(self, image, label, edge):
        h, w = image.shape[:-1]
        image = self.pad_image(image,
                               h,
                               w,
                               self.crop_size,
                               padvalue=(0.0, 0.0, 0.0))  # 최소 1024, 1024
        label = self.pad_image(label,
                               h,
                               w,
                               self.crop_size,
                               padvalue=(self.ignore_label, ))  # 255
        edge = self.pad_image(edge, h, w, self.crop_size, padvalue=(0.0, ))

        new_h, new_w = label.shape
        x = random.randint(0, new_w - self.crop_size[1])
        y = random.randint(0, new_h - self.crop_size[0])
        image = image[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        label = label[y:y + self.crop_size[0], x:x + self.crop_size[1]]
        edge = edge[y:y + self.crop_size[0], x:x + self.crop_size[1]]

        return image, label, edge

    def multi_scale_aug(self,
                        image,
                        label=None,
                        edge=None,
                        rand_scale=1,
                        rand_crop=True):
        # rand_scale: 0.5 ~ 2.1 중 랜덤한 값
        # base_size: 2048
        # 2048
        long_size = np.int(self.base_size * rand_scale + 0.5)  # 1024 ~ 4300
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size  # 1024 ~ 4300
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                edge = cv2.resize(edge, (new_w, new_h),
                                  interpolation=cv2.INTER_NEAREST)
        else:
            return image

        if rand_crop:
            image, label, edge = self.rand_crop(image, label, edge)

        return image, label, edge

    def gen_sample(self,
                   image,
                   label,
                   multi_scale=True,
                   is_flip=True,
                   edge_pad=True,
                   edge_size=4,
                   city=True):

        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((edge_size, edge_size), np.uint8)
        if edge_pad:
            edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
            edge = np.pad(edge, ((y_k_size, y_k_size), (x_k_size, x_k_size)),
                          mode='constant')
        edge = (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0

        if multi_scale:
            # 랜덤한 크기로 이미지, 레이블, 엣지 정보를 조절합니다.
            # 0 ~ 16 ->
            # 0.5 ~ 2.1
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            # (1024, 1024 전부 같음)
            image, label, edge = self.multi_scale_aug(image,
                                                      label,
                                                      edge,
                                                      rand_scale=rand_scale)

        image = self.input_transform(image, city=city)
        label = self.label_transform(label)

        # (H, W, C) -> (C, H, W)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            # horizontal flip 할지 말지 50:50
            image = image[:, :, ::flip]
            label = label[:, ::flip]
            edge = edge[:, ::flip]

        return image, label, edge

    def inference(self, config, model, image):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:  # 2
            pred = pred[config.TEST.OUTPUT_INDEX]  # 1

        pred = F.interpolate(input=pred,
                             size=size[-2:],
                             mode='bilinear',
                             align_corners=config.MODEL.ALIGN_CORNERS)

        return pred.exp()
