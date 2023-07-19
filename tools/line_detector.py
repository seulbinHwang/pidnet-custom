import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse
import datetime


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


if __name__ == '__main__':
    args = parse_args()
    dir = os.path.join(args.r, args.sub, '*' + args.t)
    print("dir:", dir)
    images_list = glob.glob(dir)
    # print("images_list:", images_list)
    current_datetime = datetime.datetime.now().strftime("%m%d_%H%M%S")
    sv_path = args.r + f'line_outputs_{current_datetime}/'
    for img_path in images_list:
        img_name = img_path.split("/")[-1]
        # Load the image
        img = cv2.imread(os.path.join(args.r, args.sub, img_name),
                         cv2.IMREAD_COLOR)

        # Convert the image to HSV color space
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the color range for white color in HSV
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([180, 20, 255])

        # Threshold the HSV image to get only white colors
        mask = cv2.inRange(img_hsv, lower_white, upper_white)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask=mask)
        # res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

        if not os.path.exists(sv_path):
            os.mkdir(sv_path)
        # Save the result image
        cv2.imwrite(sv_path + img_name, res)

        # # Convert color to RGB (from BGR)
        # res_rgb = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
        #
        # # Plot the original image and the result
        # plt.figure(figsize=(20, 10))
        # plt.subplot(1, 2, 1)
        # plt.title('Original image')
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.subplot(1, 2, 2)
        # plt.title('Detected white lines')
        # plt.imshow(res_rgb)
        # plt.show()
        #
