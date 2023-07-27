import scipy.io
import os
import cv2

# Load the .mat file
dir = './human_semantic_similarity.mat'
current_dir = os.path.dirname(__file__)
abs_dir  = os.path.abspath(os.path.join(current_dir,dir))

mat = scipy.io.loadmat(abs_dir)

# Show the keys in the .mat file
print(mat.keys())
"""
__header__
__version__
__globals__
similarity
object_names
"""

# Inspect the 'similarity' and 'object_names' arrays
similarity = mat['similarity'] # (150, 150)
object_names = mat['object_names'] # (1, 150)

print(similarity.shape, object_names.shape)

# Convert object names to list
object_names_list = [str(item[0]) for item in object_names[0]]

# # Show first few object names
print(object_names_list[:13])
#
# # Show a small part of the similarity matrix
# print(similarity[:5, :5])

# I want (x,y) coordinates of the cells with above 0.5 similarity and only visualize those cells.
post_threshold = similarity[12:13]
# TO float



