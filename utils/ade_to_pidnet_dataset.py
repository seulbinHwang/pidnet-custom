"""
- ADE20k 데이터셋에서, person/wall이 있는 dataset 만들기
    - ADE
        - training (validation) (~/Downloads/ADE20K_2021_17_01/images/ADE)
            - 폴더 뭉치 As 다 구하기
                - for A in As:
                    - 폴더 뭉치 Bs다 구하기
                        - for B in Bs:
                            - ADE_train (ADE_val) 으로 시작하고, .jpg로 끝나는 파일 전부 찾음 =JPGS 거기에, .jpg를 지우고, _seg.png를 붙여서 List[List[str, str]]을 만든다.
                                - for list in lists:
                                    - jpg, _seg.png = list
                                    - ```data/list/trainval.lst```(```data/list/val.lst```)+ 가 없으면 만든 후, 열어서, 한줄 적고 줄 엔터를 합니다.
                                    - jpg를 ```data/ADE/leftImg8bit/train/A/B```(```data/ADE/leftImg8bit/val/A/B```) 로 옮깁니다. (없으면 폴더 생성합니다.)
                                    - _seg.png를 ```data/ADE/gt_Fine/train/A/B```(```data/ADE/gt_Fine/val/A/B```)로 옮깁니다. (없으면 폴더 생성합니다.)

DATASET_NUMBER = 27566
"""
import os
from typing import List
import shutil
import traceback
import cv2
import numpy as np
from utils.get_dataset_colormap import create_ade20k_labels
from utils.get_dataset_colormap import create_ade20k_label_colormap

def copy_file(source_file, destination_folder):
    try:
        shutil.copy(source_file, destination_folder)
        # print("파일이 성공적으로 복사되었습니다.")
    except FileNotFoundError:
        print("복사할 파일을 찾을 수 없습니다.")
    except PermissionError:
        print("권한이 없어서 파일을 복사할 수 없습니다.")
    except Exception as e:
        print("파일 복사 중 오류가 발생했습니다:", str(e))
idx_to_label = create_ade20k_labels()
label_to_idx = {value: int(key) for key, value in idx_to_label.items()}

color_map = create_ade20k_label_colormap()

# person
person_rgb = color_map[label_to_idx["person"]]
bicycle_rgb = color_map[label_to_idx["bicycle"]]
# grass
grass_rgb = color_map[label_to_idx["grass"]]
# wall
wall_rgb = color_map[label_to_idx["wall"]]
# ball
ball_rgb = color_map[label_to_idx["ball"]]

# 실제로 해당 사용자의 홈 디렉토리 경로로 변환.
original_dataset_path = os.path.expanduser(
    "~/Downloads/ADE20K_2021_17_01/images/ADE")
trainval_lst_path = os.path.expanduser(
    "~/PycharmProjects/pidnet-custom/data/list/ade/trainval.lst")
train_lst_path = os.path.expanduser(
    "~/PycharmProjects/pidnet-custom/data/list/ade/train.lst")
val_lst_path = os.path.expanduser(
    "~/PycharmProjects/pidnet-custom/data/list/ade/val.lst")
# Remove all contents in lst_paths
for lst_path in [trainval_lst_path, train_lst_path, val_lst_path]:
    try:
        with open(lst_path, 'w') as lst_file:
            lst_file.truncate(0)
        print("파일 내용이 성공적으로 삭제되었습니다.")
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
    except Exception as e:
        print("파일 작성 중 오류가 발생했습니다:", str(e))
save_gtFine_dataset_path = os.path.expanduser(
    "~/PycharmProjects/pidnet-custom/data/ade/gtFine")
save_leftImg8bit_dataset_path = os.path.expanduser(
    "~/PycharmProjects/pidnet-custom/data/ade/leftImg8bit")
original_training_dataset_path = os.path.join(
    original_dataset_path,
    "training")  # "~/Downloads/ADE20K_2021_17_01/images/ADE/training"
original_validation_dataset_path = os.path.join(
    original_dataset_path,
    "validation")  # "~/Downloads/ADE20K_2021_17_01/images/ADE/validation"
# find all folders in original_training_dataset_path
original_training_folders = os.listdir(original_training_dataset_path)
# remove .DS_Store
if ".DS_Store" in original_training_folders:
    original_training_folders.remove(".DS_Store")


original_validation_folders = os.listdir(original_validation_dataset_path)
# remove .DS_Store
if ".DS_Store" in original_validation_folders:
    original_validation_folders.remove(".DS_Store")

total_images_num = 0
total_person_images_num = 0
total_bicycle_images_num = 0
total_grass_images_num = 0
total_wall_images_num = 0
total_ball_images_num = 0



for idx, original_folders in enumerate([
        original_training_folders, original_validation_folders
]):
    if idx == 0:
        category = "train"
        lst_paths = [trainval_lst_path, train_lst_path]
        # "~/Downloads/ADE20K_2021_17_01/images/ADE/training"
        original_top_dataset_path = original_training_dataset_path
    else:
        category = "val"
        lst_paths = [trainval_lst_path, val_lst_path]
        # "~/Downloads/ADE20K_2021_17_01/images/ADE/validation"
        original_top_dataset_path = original_validation_dataset_path
    # original_folders:  ['nature_landscape', 'transportation', 'urban', 'sports_and_leisure', 'unclassified', 'home_or_hotel', 'industrial', 'work_place', 'shopping_and_dining', 'cultural']
    for original_folder in original_folders:
        original_sub_folders = os.listdir(
            os.path.join(original_top_dataset_path, original_folder))
        # remove .DS_Store
        if ".DS_Store" in original_sub_folders:
            original_sub_folders.remove(".DS_Store")
        for original_sub_folder in original_sub_folders:
            # Find all files that ends with .jpg in original_sub_folder
            # Do not use List comprehension. It is too long.
            original_jpgs_and_seg_pngs = []  # List[List[str, str]]
            data_files = os.listdir(
                os.path.join(original_top_dataset_path, original_folder,
                             original_sub_folder))
            if ".DS_Store" in data_files:
                # remove .DS_Store
                data_files.remove(".DS_Store")
            for data_file in data_files:
                if data_file.endswith(".jpg"):
                    total_images_num += 1
                    jpg_file = data_file
                    seg_png_file = jpg_file.replace(".jpg", "_seg.png")
                    original_jpgs_and_seg_pngs.append(
                        [jpg_file, seg_png_file])
                    # ['ADE_train_00020252.jpg', 'ADE_train_00020252_seg.png']
                    # ['ADE_val_00000397.jpg', 'ADE_val_00000397_seg.png']
            for lst_path in lst_paths:
                # Check if lst_path exists, if not, raise FileNotFoundError.
                try:
                    with open(lst_path, 'a') as lst_file:
                        for original_jpg_and_seg_png in original_jpgs_and_seg_pngs:
                            # leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png	gtFine/train/aachen/aachen_000000_000019_gtFine_labelIds.png
                            jpg_file, seg_png_file = original_jpg_and_seg_png
                            suffix = "_leftImg8bit.jpg"
                            suffixed_jpg_file = jpg_file.replace(
                                ".jpg", suffix)
                            jpg_file_path = os.path.join(
                                "leftImg8bit", category, original_folder,
                                original_sub_folder, suffixed_jpg_file)
                            seg_jpg_file = seg_png_file.replace(
                                "_seg.png", ".jpg")
                            suffix = "_gtFine_labelIds.jpg"
                            suffixed_seg_jpg_file = seg_jpg_file.replace(
                                ".jpg", suffix)
                            seg_jpg_file_path = os.path.join(
                                "gtFine", category, original_folder,
                                original_sub_folder, suffixed_seg_jpg_file)
                            write_str = jpg_file_path + " " + seg_jpg_file_path + "\n"
                            lst_file.write(write_str)

                except FileNotFoundError:
                    print("파일을 찾을 수 없습니다.")
                    traceback.print_exc()
                except Exception as e:
                    print("파일 작성 중 오류가 발생했습니다:", str(e))
                    traceback.print_exc()
            # original_jpgs_and_seg_pngs: List[List[str, str]]
            # Copy jpg_file to save_leftImg8bit_dataset_path using copy_file function.
            # save_leftImg8bit_dataset_path: "~/PycharmProjects/pidnet-custom/data/ade/leftImg8bit"
            for original_jpg_and_seg_png in original_jpgs_and_seg_pngs:

                jpg_file, seg_png_file = original_jpg_and_seg_png
                seg_jpg_file = seg_png_file.replace(
                    "_seg.png", ".jpg")
                save_leftImg8bit_path = os.path.join(
                    save_leftImg8bit_dataset_path, category,
                    original_folder, original_sub_folder)

                if not os.path.exists(save_leftImg8bit_path):
                    os.makedirs(save_leftImg8bit_path)
                # original_top_dataset_path: "~/Downloads/ADE20K_2021_17_01/images/ADE/training"
                source_file = os.path.join(
                    original_top_dataset_path, original_folder,
                    original_sub_folder, jpg_file)
                copy_file(source_file,
                          destination_folder=save_leftImg8bit_path)
                # Change saved jpg file to _leftImg8bit.jpg
                source_file = os.path.join(
                    save_leftImg8bit_path, jpg_file)
                suffix = "_leftImg8bit.jpg"
                suffix_jpg_file = jpg_file.replace(".jpg", suffix)
                os.rename(source_file, suffix_jpg_file)
                #
                # Change seg_png_file to seg_jpg_file and save it to save_gtFine_dataset_path.
                # save_gtFine_dataset_path: "~/PycharmProjects/pidnet-custom/data/ade/gtFine"
                save_gtFine_path = os.path.join(
                    save_gtFine_dataset_path, category,
                    original_folder, original_sub_folder)
                # print("save_gtFine_path:", save_gtFine_path)
                if not os.path.exists(save_gtFine_path):
                    os.makedirs(save_gtFine_path)
                # seg_png_file = 'ADE_train_00020252_seg.png'
                source_file = os.path.join(
                    original_top_dataset_path, original_folder,
                    original_sub_folder, seg_png_file)
                copy_file(source_file,
                          destination_folder=save_gtFine_path)
                # Change saved png file to jpg file.
                source_file = os.path.join(save_gtFine_path,
                                           seg_png_file)
                suffix = "_gtFine_labelIds.jpg"
                seg_jpg_file = seg_png_file.replace(".jpg", suffix)
                rename_file = os.path.join(save_gtFine_path,
                                           seg_jpg_file)
                # Rename from png to jpg.
                os.rename(source_file, rename_file)
#                 # laod rename_file using cv2.imread
#                 img = cv2.imread(rename_file) # (1536, 2048, 3)
#                 person_matches = np.all(img == person_rgb, axis=-1) # (1536, 2048)
#                 person_num = np.sum(person_matches)
#                 if person_num > 0:
#                     total_person_images_num += 1
#                 bicycle_matches = np.all(img == bicycle_rgb, axis=-1) # (1536, 2048)
#                 bicycle_num = np.sum(bicycle_matches)
#                 if bicycle_num > 0:
#                     total_bicycle_images_num += 1
#                 grass_matches = np.all(img == grass_rgb, axis=-1) # (1536, 2048)
#                 grass_num = np.sum(grass_matches)
#                 if grass_num > 0:
#                     total_grass_images_num += 1
#                 wall_matches = np.all(img == wall_rgb, axis=-1) # (1536, 2048)
#                 wall_num = np.sum(wall_matches)
#                 if wall_num > 0:
#                     total_wall_images_num += 1
#                 ball_matches = np.all(img == ball_rgb, axis=-1) # (1536, 2048)
#                 ball_num = np.sum(ball_matches)
#                 if ball_num > 0:
#                     total_ball_images_num += 1
#
# print("total_images_num:", total_images_num)
# print("-------------------------")
# print("total_person_images_num:", total_person_images_num)
# print("total_person_images_rate:", total_person_images_num / total_images_num)
# print("-------------------------")
# print("total_bicycle_images_num:", total_bicycle_images_num)
# print("total_bicycle_images_rate:", total_bicycle_images_num / total_images_num)
# print("-------------------------")
# print("total_grass_images_num:", total_grass_images_num)
# print("total_grass_images_rate:", total_grass_images_num / total_images_num)
# print("-------------------------")
# print("total_wall_images_num:", total_wall_images_num)
# print("total_wall_images_rate:", total_wall_images_num / total_images_num)
# print("-------------------------")
# print("total_ball_images_num:", total_ball_images_num)
# print("total_ball_images_rate:", total_ball_images_num / total_images_num)



