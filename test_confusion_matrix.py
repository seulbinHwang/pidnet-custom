from utils.utils import get_confusion_matrix

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Assuming that `labels` and `preds` are your actual data
# Here we are just creating some random data for the purpose of this example
num_images = 10
height, width = 100, 100
num_classes = 3

# Generate some random binary labels
labels = torch.randint(0, num_classes, size=(num_images, height, width))

# Generate some random prediction probabilities, then take the argmax
# to simulate class predictions
pred_probs = torch.rand(num_images, num_classes, height, width)
preds = pred_probs

# Calculate confusion matrix
confusion_matrix = get_confusion_matrix(labels, preds, labels.shape, num_classes, ignore_label=255)

# Normalize confusion matrix
cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

# Display confusion matrix
plt.figure(figsize=(10, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=['class_0', 'class_1', 'class_2'])
disp.plot(include_values=True,
          cmap='viridis', ax=None, xticks_rotation='horizontal',
          values_format=None)
plt.show()
