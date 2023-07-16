import os
import glob
import datetime

from pylab import imshow
import numpy as np
import cv2
import torch
import albumentations as albu
import argparse
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model


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
  parser.add_argument(
    '--p',
    help='dir for pretrained model',
    default='../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt',
    type=str)
  parser.add_argument('--r',
                      help='root or dir for input images',
                      default='./samples/',
                      type=str)
  # resized_data
  parser.add_argument('--sub', help='sub', default='resized_data_from_video',
                      type=str)
  parser.add_argument('--t',
                      help='the format of input images (.jpg, .png, ...)',
                      default='.jpg',
                      type=str)

  args = parser.parse_args()

  return args


if __name__ == '__main__':
    args = parse_args()
    model = create_model("Unet_2020-07-20")
    model.eval()
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    dir = os.path.join(args.r, args.sub, '*' + args.t)

    print("dir:", dir)
    images_list = glob.glob(dir)
    # print("images_list:", images_list)
    current_datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")
    sv_path = args.r + f'line_outputs_{current_datetime}/'
    for img_path in images_list:
        img_name = img_path.split("/")[-1]
        path = os.path.join(args.r, args.sub, img_name)
        image = load_rgb(path)
        padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
        x = transform(image=padded_image)["image"]
        x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
        with torch.no_grad():
          prediction = model(x)[0][0]
        mask = (prediction > 0).cpu().numpy().astype(np.uint8)
        mask = unpad(mask, pads)
        if not os.path.exists(sv_path):
            os.mkdir(sv_path)
        # Save the result image
        cv2.imwrite(sv_path+img_name, mask)