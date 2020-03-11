import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
from scipy.stats import mode
import cv2
from torchvision import models
import torch
from PIL import Image

# extract image names from shuffle of images I have obtained
#training
training_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir("labelled/train/"))
training_labels_indices = list(training_labels_indices)
# testing
test_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir("labelled/test/"))
test_labels_indices = list(test_labels_indices)

# read in labels csv file
training_labels = pd.read_csv("boneage-training-dataset.csv")
#test_labels = pd.read_csv("boneage-test-dataset.csv")

# only keep entries where image is present
training_labels = training_labels.loc[training_labels["id"].isin(training_labels_indices)]
#test_labels = test_labels.loc[test_labels["Case ID"].isin(test_labels_indices)]

## TRYING TO 'MASK' OUT THE TEXT/LABELS
# load images
example = io.imread("labelled/train/1468.png")
# mask = np.zeros(example.shape)
# above_level = example > 165
# mask[np.where(above_level)] = example[above_level]
#mask = mask.astype(int)
#
# plt.imshow(mask, cmap = 'gray')
# plt.show()
#
# example[above_level] = np.median(example)
# example[above_level] = mode(example, axis = None)
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
#X_train = np.empty([training_labels.shape[0], np.prod(example.shape)])

i = 0
image_size = np.empty([training_labels.shape[0], 2])
for filename in training_labels['id'].values:
    #load image
    image = io.imread('labelled/train/'+str(filename)+'.png')
    #reshape image as done previously and store in X_train ith row
    #X_train[i,:] = image.reshape(1, np.prod(image.shape))
    image_size[i,:] = image.shape
    #update indexing variable i
    i +=1

plt.scatter(image_size[:, 0], image_size[:, 1])
plt.title("Varying image size across dataset")
plt.xlabel("'Row' pixels")
plt.ylabel("'Column' pixels")
plt.show()

from skimage.io import imread


img = imread("labelled/train/1468.png")

img_centered = img-img.mean()
img_normalised = (img-img.mean())/img.std()

fig, axs = plt.subplots(3, sharex=True)
axs[0].hist(img.flatten(), bins = 256)
axs[0].set_title('original')
axs[1].hist(img_centered.flatten(), bins = 256)
axs[1].set_title('Centered')
axs[2].hist(img_normalised.flatten(), bins = 256)
axs[2].set_title('Centered+Scaled')
plt.show()

fig, axs = plt.subplots(1, 3)
axs[0].imshow(img)
axs[0].set_title('original')
axs[1].imshow(img_centered)
axs[1].set_title('Centered')
axs[2].imshow(img_normalised)
axs[2].set_title('Centered+Scaled')
plt.show()

