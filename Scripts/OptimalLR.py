import torch
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd

import math

try:
    from Scripts.MyResNet import ResNet, BasicBlock, Bottleneck
    from Scripts.MyTransforms import *
    from Scripts.LoadImages import *
except:
    from MyResNet import ResNet, BasicBlock, Bottleneck
    from MyTransforms import *
    from LoadImages import *
print("Local modules imported")

def find_lr(init_value = 1e-8, final_value=10., beta = 0.98):
    # the bit down here comes from "maths" in Sylvain gugger's blog
    # basically we need one more batch for this whole thing to make sense
    num = len(train_loader)-1
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    # one epoch
    for data in train_loader:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,label1, _ = data
        # inputs, label1, _ = Variable(inputs), Variable(labels)
        inputs, label1 = inputs.double(), label1.double()
        
        inputs, label1 = inputs.to(device), label1.to(device)
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, label1)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
    return log_lrs, losses

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
                                CHALE(),
                                InstanceNorm(),
                                ToTensor()
                                ]),
                           batch_size=128, plot=0, save=0)

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
                          batch_size=500, plot=0, save=0)

else:
    ## LOAD DATA -- on the fly
    train_loader = getData("../labelled/train/",
                           "../boneage-training-dataset.csv",
                           transform=transforms.Compose(
                               [Rescale(256),
                                # RandomCrop(224),
                                CenterCrop(224),
                                CHALE(),
                                InstanceNorm(),
                                ToTensor()
                                ]),
                           plot=0, batch_size=8)

    test_loader = getData("../labelled/test/",
                          "../boneage-training-dataset.csv",
                          transform=transforms.Compose(
                              [Rescale(256),
                               # RandomCrop(224),
                               CenterCrop(224),
                               CHALE(),
                               InstanceNorm(),
                               ToTensor()
                               ]),
                          plot=0, batch_size=10)

print("Success! Data loaded\n")


device = torch.device("cuda" if use_cuda else "cpu")
print("You are using a "+str(device))

net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes = 1)
net = net.double()
net = net.to(device)
optimizer = optim.SGD(net.parameters(),lr=1e-1)
criterion = F.l1_loss

print("Starting learning rate optimization:")
logs, losses = find_lr()

results = {'lr':logs, 'loss':losses}
results = pd.DataFrame(results)
results.to_csv("Results/CHALEOptimLR.csv")
#plt.plot(logs[10:-5], losses[10:-5])
print("Optimization finished.")
# learner = Learner(bunch, net, metrics = mae)
# learner.lr_find()
#
# learner.recorder.plot()

