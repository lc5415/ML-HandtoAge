import pandas as pd
import os
import re
import matplotlib.pyplot as plt
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

#### histogram visualisation

RawData = getData("labelled/train",
                          "boneage-training-dataset.csv",
                          transform=transforms.Compose(
                                  [Rescale(256),
                                   # RandomCrop(224),
                                   CenterCrop(224),
                                   ToTensor(),
                                   #Normalize([0.2011], [0.1847])
                                   ]), batch_size=20, normalise=False, clahe=False, plot = 1)
