import pandas as pd
import os
import re
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use(['seaborn-colorblind'])
plt.rc('axes', axisbelow=True)
from skimage import io
import numpy as np
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

if platform.system() == 'Linux':
    data_path = 'FULLdata/'
else:
    data_path = 'labelled/train/'


# a few images

image_directory = "labelled/train"
labels_directory = "boneage-training-dataset.csv"
labels_indices = map(lambda filename: filename.split('.')[0], os.listdir(image_directory))
labels_indices = pd.DataFrame(list(labels_indices))

data_labels = pd.read_csv(labels_directory)

DATASET = HandDataset(labels_indices,
                      data_labels,
                      image_directory,
                      transform=transforms.Compose(
                                  [Rescale(256),
                                   # RandomCrop(224),
                                   CenterCrop(224),
                                   # CHALE(),
                                    InstanceNorm(),
                                   ToTensor()]))

fig1 = DATASET.plot_n_images()
# fig1.savefig('Images/ScaleCropImagesGrid.png')
fig2 = DATASET.n_histograms()
# fig2.savefig('Images/ScaleCropHistGrid.png')
#
# ####### Image resolution
#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#
#
# i = 0
# image_size = np.empty([labels_indices.shape[0], 2])
# for filename in labels_indices.iloc[:,0].values:
#     #load image
#     image = io.imread(data_path+str(filename)+'.png')
#     #reshape image as done previously and store in X_train ith row
#     #X_train[i,:] = image.reshape(1, np.prod(image.shape))
#     image_size[i,:] = image.shape
#     #update indexing variable i
#     i +=1
#
#
# ############### MAYBE JUST SAVE AND PLOT IN GGPLOTS
# ax3.scatter(image_size[:, 0], image_size[:, 1])
# ax3.set_title("Varying image size across dataset")
# ax3.set_xlabel("'Row' pixels")
# ax3.set_ylabel("'Column' pixels")
# ax3.grid()
#
# ax1 = sns.kdeplot(image_size[:, 0], bw=0.5)
#
# ax4 = sns.kdeplot(image_size[:, 1], bw=0.5)
#
# plt.show()
#
# #### histogram visualisation
#
# RawData = getData("labelled/train",
#                           "boneage-training-dataset.csv",
#                           transform=transforms.Compose(
#                                   [Rescale(256),
#                                    CenterCrop(224),
#                                    CHALE(),
#                                    InstanceNorm(),
#                                    ToTensor()]),
#                                     batch_size=20, plot = 1)

