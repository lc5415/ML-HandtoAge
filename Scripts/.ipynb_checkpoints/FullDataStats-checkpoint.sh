#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=8:mem=32gb
#PBS -N FullStats

module load anaconda3/personal

cd /rdsgpfs/general/user/lc5415/home/ML-HtoA

python << END
import Scripts.LoadImages as LoadImages

import torch
from torchvision import transforms, utils
import pandas as pd
import os, re
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import time
import torchvision.utils as utils
from Scripts.MyTransforms import Rescale, RandomCrop, ToTensor
from torchvision.transforms import Normalize
plt.rcParams['image.cmap'] = 'gray' # set default colormap to gray

st = time.time()
Fulldata = LoadImages.getData("FULLdata/training","boneage-training-dataset.csv",
        transform=transforms.Compose(
                                  [Rescale(256),
                                   RandomCrop(224),
                                   ToTensor()
                                   ]),
        batch_size="full", normalise=False, plot = 0)

mean, std = LoadImages.FullBatchStats(Fulldata)
print(mean, std)
print(time.time()-st)

END


