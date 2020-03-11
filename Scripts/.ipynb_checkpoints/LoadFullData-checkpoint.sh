#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=4:mem=8gb

module load anaconda3/personal

cd /rdsgpfs/general/user/lc5415/home/ML-HtoA

python << END
import Scripts.LoadImages as LoadImages
import Scripts.LoadImages

import torch
from torchvision import transforms, utils
import pandas as pd
import os, re
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import torchvision.utils as utils
from Scripts.MyTransforms import Rescale, RandomCrop, ToTensor
from torchvision.transforms import Normalize
plt.rcParams['image.cmap'] = 'gray' # set default colormap to gray

LoadImages.getData("FULLdata/training","boneage-training-dataset.csv",
        transform=transforms.Compose(
                                  [Rescale(256),
                                   RandomCrop(224),
                                   ToTensor()
                                   ]),
        batch_size=128, normalise=True, plot = 0, save = 1, savename = "FUULData/trainLoaded.pt")

LoadImages.getData("FULLdata/test","boneage-training-dataset.csv",
        transform=transforms.Compose(
                                  [Rescale(256),
                                   RandomCrop(224),
                                   ToTensor()
                                   ]),
        batch_size="full", normalise=True, plot = 0, save = 1, savename = "FULLData/testLoaded.pt")
END


