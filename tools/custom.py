# ------------------------------------------------------------------------------
# Written by Jiacong Xu (jiacong.xu@tamu.edu)
# ------------------------------------------------------------------------------

import glob
import argparse
import cv2
import os
import numpy as np
import models
import torch
import torch.nn.functional as F
from PIL import Image
import datetime

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

color_map = [
    #(128, 64, 128),  # road
    #(244, 35, 232),  # sidewalk
    #(70, 70, 70),  # building
    #(102, 102, 156),  # wall
    #(190, 153, 153),  # fence
    #(153, 153, 153),
    #(250, 170, 30),
    #(220, 220, 0),
    #(107, 142, 35),  # vegetation (nature)
    (152, 251, 152),  # terrain (nature) 지역
    #(70, 130, 180),  # sky
    (220, 20, 60),  # person
    #(255, 0, 0),
    #(0, 0, 142),
    #(0, 0, 70),
    #(0, 60, 100),
    #(0, 80, 100),
    #(0, 0, 230),
    #(119, 11, 32)
]


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')

    parser.add_argument('--a',
                        help='pidnet-s, pidnet-m or pidnet-l',
                        default='pidnet-l',
                        type=str)
    parser.add_argument('--c',
                        help='cityscapes pretrained or not',
                        type=bool,
                        default=True)
    parser.add_argument('--p',
                        help='dir for pretrained model',
                        default='./pretrained_models/cityscapes/best.pt',
                        type=str)
    parser.add_argument('--r',
                        help='root or dir for input images',
                        default='./samples/',
                        type=str)
    # resized_data
    parser.add_argument('--sub',
                        help='sub',
                        default='resized_data_from_video',
                        type=str)
    parser.add_argument('--t',
                        help='the format of input images (.jpg, .png, ...)',
                        default='.jpg',
                        type=str)

    args = parser.parse_args()

    return args


def input_transform(image):
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image


def load_pretrained(model, pretrained):
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {
        k[6:]: v
        for k, v in pretrained_dict.items()
        if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)
    }
    msg = 'Loaded {} parameters!'.format(len(pretrained_dict))
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    return model


def concatenate_two_images(sv_img: Image, img_path: str):
    img = Image.open(img_path)
    img = img.resize((sv_img.width, sv_img.height))
    img = Image.blend(img, sv_img, 0.5)
    return Image.fromarray(np.hstack((np.array(img), np.array(sv_img))))


# python tools/custom.py --a 'pidnet-l' --p 'pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt' --t '.png'
if __name__ == '__main__':
    args = parse_args()
    dir = os.path.join(args.r, args.sub, '*' + args.t)
    print("dir:", dir)
    images_list = glob.glob(dir)
    # print("images_list:", images_list)
    current_datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")
    sv_path = args.r + f'outputs_{current_datetime}/'
    if args.c:
        num_classes = 2
    else:
        num_classes = 11
    model = models.pidnet.get_pred_model(name=args.a, num_classes=num_classes)
    model = load_pretrained(model, args.p).cpu()
    model.eval()
    with torch.no_grad():
        for img_path in images_list:
            # print("img_path:", img_path)
            img_name = img_path.split("/")[-1]
            img = cv2.imread(os.path.join(args.r, args.sub, img_name),
                             cv2.IMREAD_COLOR)
            # img = cv2.resize(img, (720, 960), interpolation=cv2.INTER_LINEAR)
            # 1024 -> 128 (1/8배)
            sv_img = np.zeros_like(img).astype(np.uint8)
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cpu()
            # img: torch.Size([1, 3, H, W])

            pred = model(img)
            # pred: (1, class_num, 128, 256)
            pred = F.interpolate(pred,
                                 size=img.size()[-2:],
                                 mode='bilinear',
                                 align_corners=True)
            # pred: (1, class_num, 1024, 2048)

            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            # (1, class_num, 1024, 2048) -> (1, 1024, 2048) -> (1024, 2048)
            for i, color in enumerate(color_map):
                for j in range(3):
                    # 9 잔디
                    #
                    # if i not in [9, 11]:
                    #     continue
                    sv_img[:, :, j][pred == i] = color[j]
            sv_img = Image.fromarray(sv_img)
            sv_img = concatenate_two_images(sv_img, img_path)

            if not os.path.exists(sv_path):
                os.mkdir(sv_path)
            sv_img.save(sv_path + img_name)
