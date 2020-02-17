import torch.nn as nn
import LoadImages # the loading function/script I made
from MyResNet import ResNet


data = LoadImages.main()

myNet = ResNet(num_classes=1)
ResNet.train()

# follow MNIST pytorch tutorial to creat main() script