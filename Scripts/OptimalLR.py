import torch

from fastai.vision import *
from fastai.metrics import mae
import libsixel

try:
    from Scripts.MyResNet import ResNet, BasicBlock, Bottleneck
    from Scripts.MyTransforms import *
    from Scripts.LoadImages import *
except:
    from MyResNet import ResNet, BasicBlock, Bottleneck
    from MyTransforms import *
    from LoadImages import *
print("Local modules imported")

use_cuda = torch.cuda.is_available()

print("Your working directory is {}\n".format(os.getcwd()))
print("Loading data...")
if use_cuda:
    #         train_loader = test_loader = DataLoader()
    #         train_loader = torch.load("FULLdata/trainLoaded.pt")
    #         test_loader = torch.load("FULLdata/testLoaded.pt")
    train_loader = getData("FULLdata/training",
                           "boneage-training-dataset.csv",
                           transform=transforms.Compose(
                               [Rescale(256),
                                # RandomCrop(224),
                                CenterCrop(224),
                                ToTensor()
                                ]),
                           batch_size=128, normalise=True, plot=0, save=0)

    test_loader = getData("FULLdata/test",
                          "boneage-training-dataset.csv",
                          transform=transforms.Compose(
                              [Rescale(256),
                               # RandomCrop(224),
                               CenterCrop(224),
                               ToTensor()
                               ]),
                          batch_size=500, normalise=True, plot=0, save=0)

else:
    ## LOAD DATA -- on the fly
    train_loader = getData("../labelled/train/",
                           "../boneage-training-dataset.csv",
                           transform=transforms.Compose(
                               [Rescale(256),
                                # RandomCrop(224),
                                CenterCrop(224),
                                ToTensor()
                                ]),
                           plot=0, batch_size=128)

    test_loader = getData("../labelled/test/",
                          "../boneage-training-dataset.csv",
                          transform=transforms.Compose(
                              [Rescale(256),
                               # RandomCrop(224),
                               CenterCrop(224),
                               ToTensor()
                               ]),
                          plot=0, batch_size=10)

print("Success! Data loaded\n")

net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes = 1)
net = net.double()
bunch = DataBunch(train_dl= train_loader, valid_dl = test_loader)

learner = Learner(bunch, net, metrics = mae)
learner.lr_find()

learner.recorder.plot()

