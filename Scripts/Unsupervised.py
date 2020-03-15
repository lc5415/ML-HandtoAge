from sklearn.cluster import KMeans
from torch import cuda
import numpy as np
import os
import pandas as pd
from skimage import io

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
img_path = "labelled/train/" # "../FULLdata/training/" #

training_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir(img_path))
training_labels_indices = list(training_labels_indices)

use_cuda = cuda.is_available()

# preprocessing images
if use_cuda:
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
                           batch_size=args.batch_size, plot=0, save=0)

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
                          batch_size=args.test_batch_size, plot=0, save=0)

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

#
# # Initialise X_train as empty array of shape:
# # (number of rows of TrainData = 600)x(number of elements in flatten vector (32*32*3))
# # to hold the training features,
# X_train = np.empty([len(training_labels_indices), 224*224])
# y_train = np.empty([len(train_loader)*train_loader.batch_size])
# # iterate through the filenames in the labelling list so that
# # Xtrain and labels are in same order
# i = 0  # indexing variable i
# for batch_id, (image):
#     # load image
#     image = imageio.imread(img_path+ filename + '.jpg')
#     # reshape image as done previously and store in X_train ith row
#     X_train[i, :] = image.reshape(1, np.prod(image.shape))
#     # update indexing variable i
#     i += 1
#
# # get labels from TrainData dataframe, this are in same order as X_train because of the way
# # X_train was created
# y_train = TrainData.iloc[:, 1]