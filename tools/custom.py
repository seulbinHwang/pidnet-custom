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


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')

    parser.add_argument('--network_name',
                        help='pidnet-s, pidnet-m or pidnet-l',
                        default='pidnet-l',
                        type=str)
    parser.add_argument('--use_cityscapes_pretrained',
                        help='cityscapes pretrained or not',
                        type=bool,
                        default=True)
    # number of classes
    parser.add_argument('--num_classes',
                        help='number of classes',
                        default=3,
                        type=int)
    parser.add_argument('--pretrained_model_directory',
                        help='dir for pretrained model',
                        default='./pretrained_models/cityscapes/best_3_0727.pt',
                        type=str)
    parser.add_argument('--image_root_directory',
                        help='root or dir for input images',
                        default='./samples/',
                        type=str)
    # resized_data
    parser.add_argument('--image_directory',
                        help='image_directory',
                        default='raw_data_from_video',
                        type=str)
    parser.add_argument('--image_format',
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


def load_pretrained(model, pretrained_directory):
    pretrained_dict = torch.load(pretrained_directory, map_location='cpu')
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


def concatenate_two_images(segment_result_image: Image, img_path: str):
    image_with_result = Image.open(img_path)
    image_with_result = image_with_result.resize(
        (segment_result_image.width, segment_result_image.height))
    image_with_result = Image.blend(image_with_result, segment_result_image,
                                    0.5)
    return Image.fromarray(
        np.hstack(
            (np.array(image_with_result), np.array(segment_result_image))))


# python -m tools.custom --network_name 'pidnet-l' --pretrained_model_directory 'pretrained_models/cityscapes/best.pt' --image_format '.jpg'
if __name__ == '__main__':
    args = parse_args()
    images_directories = os.path.join(args.image_root_directory,
                                      args.image_directory,
                                      '*' + args.image_format)
    # image_root_directory: ./samples/
    # image_directory: resized_data_from_video
    # image_format: .jpg
    # images_directories:/samples/resized_data_from_video/*.jpg
    images_list = glob.glob(images_directories)
    current_datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")
    save_path = args.image_root_directory + f'outputs_{current_datetime}/'
    # network_name: pidnet-l
    model = models.pidnet.get_pred_model(name=args.network_name,
                                         num_classes=args.num_classes)
    if args.num_classes == 2:
        color_map = [
            # (128, 64, 128),  # road
            # (244, 35, 232),  # sidewalk
            # (70, 70, 70),  # building
            # (102, 102, 156),  # wall
            # (190, 153, 153),  # fence
            # (153, 153, 153),
            # (250, 170, 30),
            # (220, 220, 0),
            # (107, 142, 35),  # vegetation (nature)
            (152, 251, 152),  # terrain (nature) 지역
            # (70, 130, 180),  # sky
            (220, 20, 60),  # person
            # (123, 123, 243), ########
            # (255, 0, 0),
            # (0, 0, 142),
            # (0, 0, 70),
            # (0, 60, 100),
            # (0, 80, 100),
            # (0, 0, 230),
            # (119, 11, 32)
        ]
    else:
        # terrain & person & sky
        color_map = [
            (152, 251, 152),
            (220, 20, 60),
            (70, 130, 180),
        ]
    model = load_pretrained(model, args.pretrained_model_directory).cpu()
    model.eval()
    with torch.no_grad():
        for img_path in images_list:
            # print("img_path:", img_path)
            img_name = img_path.split("/")[-1]
            original_image = cv2.imread(
                os.path.join(args.image_root_directory, args.image_directory,
                             img_name), cv2.IMREAD_COLOR)
            # original_image = cv2.resize(original_image, (1024, 1024),
            #                             interpolation=cv2.INTER_LINEAR)
            # 1024 -> 128 (1/8배)
            segment_result_image = np.zeros_like(original_image).astype(
                np.uint8)
            original_image = input_transform(original_image)
            original_image = original_image.transpose((2, 0, 1)).copy()
            original_image = torch.from_numpy(original_image).unsqueeze(0).cpu()
            # original_image: torch.Size([1, 3, H, W])

            pred = model(original_image)
            # pred: (1, class_num, 128, 256)
            pred = F.interpolate(pred,
                                 size=original_image.size()[-2:],
                                 mode='bilinear',
                                 align_corners=True)
            # pred: (1, class_num, 1024, 2048)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            # (1, class_num, 1024, 2048) -> (1, 1024, 2048) -> (1024, 2048)
            for color_idx, color in enumerate(color_map):
                for rgb_channel_idx in range(3):
                    segment_result_image[:, :, rgb_channel_idx][
                        pred == color_idx] = color[rgb_channel_idx]
            segment_result_image = Image.fromarray(segment_result_image)
            save_images = concatenate_two_images(segment_result_image, img_path)

            if not os.path.exists(save_path):
                os.mkdir(save_path)
            save_images.save(save_path + img_name)
    """
    - custom.py의 코드로 인해, 성능이 잘 안나오는 것 처럼 보이는 것은 아닐까?
        - 아닌듯 
    - cityscape test dataset에서는 잘 학습되었는데, custom dataset에서 덜 잘되는 것은 아닐까?
        - cityscape test dataset 에서의 결과?
            - 2
                - 추가 확인 필요
            - 3
                - 매우 잘 학습된 것 같음.

        - custom dataset 에서의 결과?
            - 2
            - 3
        - cityscape test dataset 에서 잘 되었다면, custom dataset과 어떤 차이가 있는지 고민해보자.
        - cityscape test / custom 차이가 별로 없다면, test dataset에서 어떤 부분이 잘 학습 안되었는지 고민해보자.


    """
