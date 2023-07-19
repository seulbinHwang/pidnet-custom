# train.py를 가져와서, cityscape dataset을 돌면서,
# label에 사람이 있으면, 해당 사진을 특정 폴더에 저장한다.
import argparse
import os
from configs import config
from configs import update_config
from typing import List, Dict
import cv2
from PIL import Image
import datasets


def parse_args():
    # python tools/train.py --cfg configs/cityscapes/pidnet_small_cityscapes.yaml GPUS (0,1) TRAIN.BATCH_SIZE_PER_GPU 6
    parser = argparse.ArgumentParser(description='Train segmentation network')

    parser.add_argument(
        '--cfg',
        help='experiment configure file name',
        default="configs/cityscapes/pidnet_small_cityscapes.yaml",
        type=str)
    parser.add_argument('--seed', type=int, default=304)
    parser.add_argument('--fine_tune', type=bool, default=False)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


ignore_label = 255
label_mapping = {
    -1: ignore_label,  #
    0: ignore_label,
    1: ignore_label,
    2: ignore_label,
    3: ignore_label,
    4: ignore_label,
    5: ignore_label,
    6: ignore_label,  # ground
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
    21: 10,  # vegetation
    22: 9,  # terrain
    23: ignore_label,  # sky
    24: ignore_label,  # person
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


def convert_label(label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in label_mapping.items():
            label[temp == k] = v
    return label


def main():
    print("main")
    args = parse_args()
    person_image_save_path = os.path.join(config.DATASET.ROOT, 'person')
    terrain_image_save_path = os.path.join(config.DATASET.ROOT, 'terrain')
    print("person_image_save_path: ", person_image_save_path)
    print("terrain_image_save_path: ", terrain_image_save_path)
    if not os.path.exists(person_image_save_path):
        os.makedirs(person_image_save_path)
    if not os.path.exists(terrain_image_save_path):
        os.makedirs(terrain_image_save_path)

    root = config.DATASET.ROOT
    crop_size = (config.TRAIN.IMAGE_SIZE[1], config.TRAIN.IMAGE_SIZE[0])
    dataset_name = 'datasets.' + config.DATASET.DATASET
    print("dataset_name: ", dataset_name)  #  datasets.cityscapes
    # NameError: name 'datasets' is not defined
    train_dataset = eval(dataset_name)(
        root=root,  # data/
        list_path=config.DATASET.TRAIN_SET,  # list/cityscapes/train.lst
        num_classes=config.DATASET.NUM_CLASSES,  # 19
        multi_scale=config.TRAIN.MULTI_SCALE,  # True
        flip=config.TRAIN.FLIP,  # True
        ignore_label=config.TRAIN.IGNORE_LABEL,  # 255
        base_size=config.TRAIN.BASE_SIZE,  # 2048
        crop_size=crop_size,  # (1024, 1024)
        scale_factor=config.TRAIN.SCALE_FACTOR)  # 16

    files: List[Dict[str, str]] = train_dataset.files
    not_count = 0
    person_count = 0
    terrain_count = 0
    vegetation_count = 0
    print("files: ", files)
    # files 의 총 파일 수 구하기.
    files_number = len(files)

    for idx, item in enumerate(files):
        label = cv2.imread(os.path.join(root, 'cityscapes', item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = convert_label(label)
        person_exists = (label == 11).sum() > 0
        vegetation_exists = (label == 10).sum() > 0
        terrain_exists = (label == 9).sum() > 0
        if person_exists or terrain_exists:
            image = cv2.imread(os.path.join(root, 'cityscapes', item["img"]),
                               cv2.IMREAD_COLOR)
            # to numpy
            if person_exists:
                person_count += 1
                print(f"person 있어요!_{person_count}/{idx}/{files_number}")
                # print("person_exists: ", person_exists)
                # mix image and label.

            if terrain_exists:
                terrain_count += 1
                print(f"terrain 있어요!_{terrain_count}/{idx}/{files_number}")
                # print("terrain_exists: ", terrain_exists)
                # cv2.imwrite(os.path.join(terrain_image_save_path, item["name"] + '.jpg'),
                #             image)
                # cv2.imwrite(os.path.join(terrain_image_save_path,
                #                          item["name"] + '_label.jpg'),
                #             label)
            if vegetation_exists:
                vegetation_count += 1
                print(
                    f"vegetation 있어요!_{vegetation_count}/{idx}/{files_number}")
            cv2.imwrite(
                os.path.join(terrain_image_save_path, item["name"] + '.jpg'),
                image)
            cv2.imwrite(
                os.path.join(terrain_image_save_path,
                             item["name"] + '_label.jpg'), label)
        else:
            not_count += 1
            print(f"없어요!_{not_count}/{idx}/{files_number}")


# person: 78%
# terrain: 56%
if __name__ == '__main__':
    main()
