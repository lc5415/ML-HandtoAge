import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import cv2
from torchvision import models
import torch
from PIL import Image

# extract image names from shuffle of images I have obtained
#training
training_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir("toy_training/"))
training_labels_indices = list(training_labels_indices)
# testing
test_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir("toy_test/"))
test_labels_indices = list(test_labels_indices)

# read in labels csv file
training_labels = pd.read_csv("boneage-training-dataset.csv")
test_labels = pd.read_csv("boneage-test-dataset.csv")

# only keep entries where image is present
training_labels = training_labels.loc[training_labels["id"].isin(training_labels_indices)]
test_labels = test_labels.loc[test_labels["Case ID"].isin(test_labels_indices)]

## TRYING TO 'MASK' OUT THE TEXT/LABELS
# load images
example = io.imread("toy_training/1468.png")
# mask = np.zeros(example.shape)
# above_level = example > 165
# mask[np.where(above_level)] = example[above_level]
# #mask = mask.astype(int)
#
# plt.imshow(mask, cmap = 'gray')
# plt.show()
#
# plt.imshow(example, cmap = 'gray')
# plt.show()
#
# mask2 = cv2.threshold(example, 165, 255, cv2.THRESH_BINARY)[1]
# plt.imshow(mask2, cmap = 'gray')
# plt.show()
#
# dst = cv2.inpaint(example, mask2, 7, cv2.INPAINT_NS)
# plt.imshow(dst, cmap = 'gray')
# plt.show()


# Segmentation network from torch
# fcn = models.segmentation.fcn_resnet101(pretrained=True).eval()

## Loading images as X's
X_train = np.empty([training_labels.shape[0], np.prod(example.shape)])
i = 0
for filename in training_labels['id'].values:
    #load image
    image = Image.open('toy_training/'+str(filename)+'.png')
    #reshape image as done previously and store in X_train ith row
    X_train[i,:] = image.reshape(1, np.prod(image.shape))
    #update indexing variable i
    i +=1



