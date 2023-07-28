import json
import numpy as np
from PIL import Image
import os
# List of pixel values to check for
pixel_values = [10, 13, 120]

# Initialize counters for each pixel value
counters = {value: 0 for value in pixel_values}


# Function to check if an image contains any of the specific pixel values
def image_contains_any_pixel(fpath_segm, pixel_values):
    """

    Args:
        fpath_segm: "ADEChallengeData2016/annotations/training/ADE_train_00000001.png"
        pixel_values: [2, 4, 13]

    Returns:

    """
    # Load the image
    img = Image.open(fpath_segm)

    # Convert the image to grayscale and then to a numpy array
    img_gray = img.convert('L')
    img_np = np.array(img_gray)

    # Check for the presence of the pixel values
    pixel_present = np.isin(img_np, pixel_values)

    # Update the counters
    for value in pixel_values:
        if np.isin(img_np, value).any():
            counters[value] += 1

    return pixel_present.any()


# Load the lines from the training.odgt file
raw_data_list_path = "data/list/ade"
raw_data_path = "data/ade"
current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.dirname(current_path)
data_list_path = os.path.join(parent_path, raw_data_list_path)
data_path = os.path.join(parent_path, raw_data_path)
training_path = f"{data_list_path}/validation.odgt"


with open(training_path, 'r') as f:
    lines = f.readlines()
original_total_images = len(lines)

filtered_lines = []
# Check each line in the file
for idx, line in enumerate(lines):
    print(f"Processing image {idx + 1} of {original_total_images}")
    raw_fpath_segm = json.loads(line)['fpath_segm']
    # "ADEChallengeData2016/annotations/training/ADE_train_00000001.png"
    fpath_segm = os.path.join(data_path, raw_fpath_segm)
    # Check if the image contains any of the pixel values
    if image_contains_any_pixel(fpath_segm, pixel_values):
        filtered_lines.append(line)

# Write the filtered lines to the new file
output_path = f"{data_list_path}/extracted_validation.odgt"
with open(output_path, 'w') as f:
    f.writelines(filtered_lines)

# Print the results
extracted_total_images = len(filtered_lines)
print("Total images in original file:", original_total_images)
print("Total images in extracted file:", extracted_total_images)
for value, count in counters.items():
    percentage = (count / original_total_images) * 100
    print(
        f"Pixel value {value} is present in {count} images ({percentage:.2f}% of total images)")

"""
training
총 이미지 수: 20210
잔디/사람/공이 있는 이미지 수: 6906
    잔디: 2421 12%
    사람: 5069 25%
    공: 136 0.67%
    
Validation
총 이미지 수: 2000
잔디/사람/공이 있는 이미지 수: 706
    잔디: 227 11.35%
    사람: 532 26.60%
    공: 25 1.25%
"""