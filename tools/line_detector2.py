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

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Apply color threshold to keep only white pixels
        lower = np.array([200, 200, 200],
                         dtype="uint8")  # Lower boundary for white color
        upper = np.array([255, 255, 255],
                         dtype="uint8")  # Upper boundary for white color
        mask = cv2.inRange(img, lower, upper)
        output = cv2.bitwise_and(img, img, mask=mask)

        # Convert the output image to grayscale
        output_gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

        # Apply Hough line transformation
        lines = cv2.HoughLinesP(output_gray,
                                1,
                                np.pi / 180,
                                100,
                                minLineLength=100,
                                maxLineGap=10)

        # Draw the lines on the original image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Convert color to RGB for plotting
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if not os.path.exists(sv_path):
            os.mkdir(sv_path)
        # Save the result image
        cv2.imwrite(sv_path + img_name, img_rgb)

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
