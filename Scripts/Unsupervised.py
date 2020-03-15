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
import platform

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
                                InstanceNorm(),
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
                               InstanceNorm(),
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
                                InstanceNorm(),
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
                               InstanceNorm(),
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


kmeans = KMeans(n_clusters = 2).fit(X_train)
y_predict = kmeans.predict(X_test)

plt.scatter(X_train[:,0], X_train[:,1], c = y_train, cmap = 'viridis')
plt.show()

## PCA for to decorrelate data and then do kmeans
pca_plot = PCA()
pca_plot.fit(X_train)
comp_to_keep = np.sum(np.cumsum(pca_plot.explained_variance_ratio_)<0.9)
pca_plot = PCA(n_components = comp_to_keep)
pca_plot.fit(X_train)
X_proj = pca_plot.transform(X_train)
kmeans2 = KMeans(n_clusters = 2).fit(X_proj)

y_train_str = ["Male" if np.array(kmeans.labels_[i]) == 1 else "Female" for i in range(len(kmeans2.labels_))]

data = (X_proj[y_train==1,:], X_proj[y_train==0,:])
colors = ("darkblue", "lightblue")
groups = ("Male", "Female")

fig, ax = plt.subplots()

for data, color, group in zip(data,colors, groups):
    x = data
    ax.scatter(x[:,0], x[:,1], c = color, label = group)
plt.legend()
plt.title("Projection of points onto PCA top-2 components")
plt.grid()
plt.savefig("Images/PCAProject.png")
plt.show()
print("Components kept for 90% explained variance: {}".format(comp_to_keep))
########## Second plot
data = (X_proj[kmeans2.labels_==1,:], X_proj[kmeans2.labels_==0,:])
colors = ("darkblue", "lightblue")
groups = ("Male", "Female")

fig, ax = plt.subplots()

for data, color, group in zip(data,colors, groups):
    x = data
    ax.scatter(x[:,0], x[:,1], c = color, label = group)
plt.legend()
plt.grid()
plt.title("Clustering by K-means on PCA-projected data")
plt.savefig("Images/KmeansCluster.png")
plt.show()

##PCA for image compression down here

# preprocessing images
if platform.system() == 'Linux':
    train_loader = getData("FULLdata/training",
                           "boneage-training-dataset.csv",
                           transform=transforms.Compose(
                               [Rescale(1024),
                                # RandomCrop(224),
                                CenterCrop(896),
                                CHALE(),
                                InstanceNorm(),
                                ToTensor()
                                ]),
                           batch_size=args.batch_size, plot=0, save=0)

    test_loader = getData("FULLdata/test",
                          "boneage-training-dataset.csv",
                          transform=transforms.Compose(
                              [Rescale(1024),
                               # RandomCrop(224),
                               CenterCrop(896),
                               CHALE(),
                               InstanceNorm(),
                               ToTensor()
                               ]),
                          batch_size=args.test_batch_size, plot=0, save=0)

else:
    ## LOAD DATA -- on the fly
    train_loader = getData("labelled/train/",
                           "boneage-training-dataset.csv",
                           transform=transforms.Compose(
                               [Rescale(1024),
                                # RandomCrop(224),
                                CenterCrop(896),
                                CHALE(),
                                InstanceNorm(),
                                ToTensor()
                                ]),
                           plot=0, batch_size="full")

    test_loader = getData("labelled/test/",
                          "boneage-training-dataset.csv",
                          transform=transforms.Compose(
                              [Rescale(1024),
                               # RandomCrop(224),
                               CenterCrop(896),
                               CHALE(),
                               InstanceNorm(),
                               ToTensor()
                               ]),
                          plot=0, batch_size="full")

numTrain = len(train_loader)*train_loader.batch_size
# Initialise X_train as empty array of shape:
# (number of rows of TrainData = 600)x(number of elements in flatten vector (32*32*3))
# to hold the training features,
X_train = np.empty([numTrain, 896*896])
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
X_test = np.empty([numTest, 896*896])
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

print("Compressing images through PCA")
pca_compress = PCA()#(n_components = 224*224)
print("Compression finished")
pca_compress.fit(X_train)

X_compressed = pca_compress.transform(X_train)
np.savetxt("Results/CompressedImages.csv", X_compressed, delimiter=",")
# plt.imshow(X_compressed[0,:144].reshape(12,12))
# plt.show()