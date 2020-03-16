from sklearn.cluster import KMeans
from torch import cuda
import numpy as np
import os
import pandas as pd
from skimage import io
from sklearn.metrics import confusion_matrix
import torch
from sklearn.decomposition import PCA, KernelPCA
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import platform
import cv2

try:
    from Scripts.MyResNet import ResNet, BasicBlock, Bottleneck
    from Scripts.MyTransforms import *
    from Scripts.LoadImages import *
except:
    from MyResNet import ResNet, BasicBlock, Bottleneck
    from MyTransforms import *
    from LoadImages import *
print("Local modules imported")

training_labels = pd.read_csv("boneage-training-dataset.csv")


if platform.system() == 'Linux':
    img_path = "FULLdata/training/"
else:
    img_path = "labelled/train/"  # "../FULLdata/training/" #


training_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir(img_path))
training_labels_indices = list(training_labels_indices)

use_cuda = cuda.is_available()

print("You are using a {}".format("cuda" if use_cuda else "cpu"))

# preprocessing images
if platform.system() == 'Linux':
    train_loader = getData("FULLdata/training",
                           "boneage-training-dataset.csv",
                           transform=transforms.Compose(
                               [Rescale(256),
                                # RandomCrop(224),
                                CenterCrop(224),
                                CHALE(),
                                # InstanceNorm(),
                                ToTensor()
                                ]),
                           batch_size="full", plot=0, save=0)

    test_loader = getData("FULLdata/test",
                          "boneage-training-dataset.csv",
                          transform=transforms.Compose(
                              [Rescale(256),
                               # RandomCrop(224),
                               CenterCrop(224),
                               CHALE(),
                               # InstanceNorm(),
                               ToTensor()
                               ]),
                          batch_size="full", plot=0, save=0)

else:
    ## LOAD DATA -- on the fly
    train_loader = getData("labelled/train/",
                           "boneage-training-dataset.csv",
                           transform=transforms.Compose(
                               [Rescale(256),
                                # RandomCrop(224),
                                CenterCrop(224),
                                CHALE(),
                                # InstanceNorm(),
                                ToTensor()
                                ]),
                           plot=0, batch_size="full")

    test_loader = getData("labelled/test/",
                          "boneage-training-dataset.csv",
                          transform=transforms.Compose(
                              [Rescale(256),
                               # RandomCrop(224),
                               CenterCrop(224),
                               CHALE(),
                               # InstanceNorm(),
                               ToTensor()
                               ]),
                          plot=0, batch_size="full")

numTrain = len(train_loader)*train_loader.batch_size
# Initialise X_train as empty array of shape:
# (number of rows of TrainData = 600)x(number of elements in flatten vector (32*32*3))
# to hold the training features,
X_train = np.empty([numTrain, 224*224])
y_train = np.empty([numTrain])
# iterate through the filenames in the labelling list so that
# Xtrain and labels are in same order
i = 0  # indexing variable i
for batch_id, (batch, _, gender) in enumerate(train_loader):
    # reshape image as done previously and store in X_train ith row
    for image in range(batch.shape[0]):
        X_train[i, :] = batch[image].flatten()
        y_train[i] = gender[image]
        # update indexing variable i
        i += 1

numTest = len(test_loader)*test_loader.batch_size
X_test = np.empty([numTest, 224*224])
y_test = np.empty([numTest])
# iterate through the filenames in the labelling list so that
# Xtrain and labels are in same order
i = 0  # indexing variable i
for batch_id, (batch, _, gender) in enumerate(test_loader):
    # reshape image as done previously and store in X_train ith row
    for image in range(batch.shape[0]):
        X_test[i, :] = batch[image].flatten()
        y_test[i] = gender[image]
        # update indexing variable i
        i += 1

print("Images loaded.")

img = X_train[1, :].reshape((224, 224))
plt.imshow(img)
plt.show()
h, w = img.shape
inertia = []
fig = plt.figure(figsize = (15,15))
img_index = 1
for cl in range(2,6):

    kmeans_image = KMeans(n_clusters = cl)

    kmeans_image.fit(img.reshape(-1,1))

    cluster_centers = kmeans_image.cluster_centers_
    cluster_labels = kmeans_image.labels_

    plt.subplot(2,2, img_index)
    plt.axis('off')
    plt.imshow(cluster_centers[cluster_labels].reshape(h,w))
    inertia.append((cl, kmeans_image.inertia_))
    img_index+=1

# plt.suptitle("Image segmentation for a single image\nfor increasing number of clusters")
plt.savefig("Images/imgsegmentSingle.png")
plt.show()

# for cl, score in inertia:
#     plt.scatter(cl, np.log10(score), c = "black")
# plt.grid()
# plt.xlabel("Number of cluster K")
# plt.ylabel("Inertia (SSD of samples to centroid)")
# plt.savefig("Images/KmeansImgSegmentScores.png")
# plt.show()

#
# ## for several images
# fig = plt.figure()
# gs1 = gridspec.GridSpec(5, 5)
# gs1.update(wspace=0, hspace=0.01)
# img_index = 0
# img_selected = np.random.randint(0, 100, 5)
# for cl in range(2, 7):
#
#     for rand_img in img_selected:
#         img = X_train[rand_img, :].reshape((224, 224))
#
#         kmeans_image = KMeans(n_clusters=cl)
#
#         kmeans_image.fit(img.reshape(-1, 1))
#
#         cluster_centers = kmeans_image.cluster_centers_
#         cluster_labels = kmeans_image.labels_
#
#         plt.subplot(gs1[img_index])
#         plt.axis('off')
#         plt.imshow(cluster_centers[cluster_labels].reshape(h, w))
#         inertia.append((cl, kmeans_image.inertia_))
#         img_index += 1
#
# plt.suptitle("Image segmentation for several images\nfor increasing number of clusters")
# plt.savefig("Images/imgsegmentMulti.png")
# plt.show()