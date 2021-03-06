import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CyclicLR
from torchvision import datasets, transforms # maybe will use this in the future
import sys
from torch.utils.data import DataLoader
from torchsummary import summary


try:
    from Scripts.MyResNet import ResNet, BasicBlock, Bottleneck
    from Scripts.MyTransforms import *
    from Scripts.LoadImages import *
except:
    from MyResNet import ResNet, BasicBlock, Bottleneck
    from MyTransforms import *
    from LoadImages import *
print("Local modules imported")



import visdom
import numpy as np
import pandas as pd


def train(args, net, device, train_loader, optimizer, epoch):
    net.train()
    train_loss = 0
    lossPerBatch = pd.DataFrame(columns=["epoch","batchNum", "Train_loss"])
    #for batch_idx, batch in enumerate(train_loader):
    for batch_idx, (data, target, _) in enumerate(train_loader):
        # get the inputs in correct format and pass them onto CPU/GPU
        # data, target = batch['image'].to(device), batch['age'].to(device) # dictionary version

        data = data.to(device)
        target = target.to(device)

        data = data.double()
        target = target.double()
        # initialize the parameters at 0
        optimizer.zero_grad()

        output = net(data)
        # compute loss
        ## change code below from -- we don't wanna be using NLL loss for regression
        # -- loss = F.nll_loss(output, target)
        ## to
        # the line below calculates average loss over batch
        # loss = F.mse_loss(output, target)

        # MAE instead of mse per batch
        loss = F.l1_loss(output, target)
        # we add up the SSE here
        train_loss += F.l1_loss(output, target, reduction = "sum").item() # same as for test set
        loss.backward()
        optimizer.step()
        
        # the train epoch message will only be printed when the batch_id module the log_interval
        # argument is equal to 0 (log interval = 10). If the number of batches is lower than 10, 
        # it will never print anything
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * train_loader.batch_sampler.batch_size,
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
        lossPerBatch.loc[len(lossPerBatch)] = [epoch, batch_idx, loss.item()]
        #we return the averaged train loss over all the batches over all the samples, this is actually 
        # like the mean. train loss is SSE, the thing we are dividing by is N.
        
        # we would prefer to output RMSE to have everything on same scale
    return train_loss/(np.floor(len(train_loader.dataset) / train_loader.batch_size) * train_loader.batch_size), lossPerBatch

def test(args, net, device, test_loader):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        # net evaulation is only performed in one batch
        # for batch_idx, batch in enumerate(test_loader):
        for batch_idx, (data, target, _) in enumerate(test_loader):
            # get the inputs in correct format and pass them onto CPU/GPU
            # data, target = batch['image'].to(device), batch['age'].to(device) # dictionary version
            data = data.to(device)
            target = target.to(device)

            data = data.double()
            target = target.double()
            output = net(data)
            # this format of mse loss compute the SSE instead of the MSE
            #test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss

            # Computing l1 loss instead of mse to have loss value in relevant units
            test_loss += F.l1_loss(output, target, reduction='sum').item()  # sum up batch loss

            # ---- This is for classification
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
    
    # this now computes the MSE
    test_loss /= (np.floor(len(test_loader.dataset)/args.test_batch_size)*args.test_batch_size)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Hand-to-Age ML Project')


    ## my code does not really take this into account
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')

    # nor does it takes this into account
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')

    ## will likely need to change this
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 14)')

    ## may need to do CV for this
    parser.add_argument('--lr', type=float, default=0.4, metavar='LR',
                        help='learning rate (default: 0.4)')

    ## may need to do CV for this
    parser.add_argument('--gamma', type=float, default=0.1, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    ## may want to do this when submitting multi-node jobs for cross-val
    parser.add_argument('--save-net', action='store_true', default=False,
                        help='For Saving the current net')
    parser.add_argument('--step-size', type = int, default = 1, metavar = 'SS',
                       help = 'StepLR scheduler step')
    ######### CUSTOM args
    parser.add_argument('--arch', type = int, default = 1,
                           help = 'ResNet architecture choice')
    parser.add_argument('-wd','--weight-decay', type = float, default = 0.01,
                           help = 'Weight decay (L2 norm, default: 0.01)')
    parser.add_argument('-rf', '--resfile', default='Results', help = "Results filename")
    args = parser.parse_args()
    
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("You are using a "+str(device))

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print(args)

    # trainloader has a batch size of 128 and testLoader has full size (2611 images)
    print("Your working directory is {}\n".format(os.getcwd()))
    print("Loading data...")
    if use_cuda:
        train_loader = getData("FULLdata/training",
                               "boneage-training-dataset.csv",
                               transform = transforms.Compose(
                                  [Rescale(256),
                                    # RandomCrop(224),
                                   CenterCrop(224),
                                   CHALE(),
                                   InstanceNorm(),
                                   ToTensor()
                                   ]),
                               batch_size=args.batch_size, plot = 0, save = 0)

        test_loader = getData("FULLdata/test",
                              "boneage-training-dataset.csv",
                               transform=transforms.Compose(
                                   [Rescale(256),
                                    #RandomCrop(224),
                                    CenterCrop(224),
                                    CHALE(),
                                    InstanceNorm(),
                                    ToTensor()
                                    ]),
                               batch_size=args.test_batch_size, plot = 0, save = 0)
        
    else:
    ## LOAD DATA -- on the fly
        train_loader = getData("labelled/train/",
                               "boneage-training-dataset.csv",
                               transform = transforms.Compose(
                                  [Rescale(256),
                                   #RandomCrop(224),
                                   CenterCrop(224),
                                   CHALE(),
                                   InstanceNorm(),
                                   ToTensor()
                                   ]),
                               plot = 0, batch_size = 8)

        test_loader = getData("labelled/test/",
                              "boneage-training-dataset.csv",
                               transform=transforms.Compose(
                                   [Rescale(256),
                                    #RandomCrop(224),
                                    CenterCrop(224),
                                    CHALE(),
                                    InstanceNorm(),
                                    ToTensor()
                                    ]),
                               plot=0, batch_size="full")

    print("Success! Data loaded\n")
    print("Setting up network's architecture...")
    architectures = [(BasicBlock, [2, 2, 2, 2]),
                     (BasicBlock, [3, 4, 6, 3]),
                     (Bottleneck, [3, 4, 6, 3]),
                     (Bottleneck, [3, 4, 23, 3])]

    #################################################################
    #                                                               #
    #        From 1st experiment we are choosing ResNet1            #
    #                                                               #
    #################################################################

    # here is where the input from the command line comes in
    chosenArch = architectures[args.arch-1]
    # set architecture from bash script call
    net = ResNet(chosenArch[0], chosenArch[1], num_classes=1)
    
    net = net.to(device)
    print(summary(net, input_size=(1, 224, 224)))
    net = net.double()
    
    ############ TRYING PARALLEL GPU WORK #####################
#     if torch.cuda.device_count() > 1:
#       print("Let's use", torch.cuda.device_count(), "GPUs!")
#       # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#       net = torch.nn.DataParallel(net)
    ##############i don't think this was worth it###################

    print("You have loaded a ResNet with {} blocks and {} layers".format(str(chosenArch[0]), str(chosenArch[1])))
    ####################### LEFT IT HERE #################

#     optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay = args.weight_decay, momentum = 0.9)
    optimizer = optim.Adam(net.parameters(), lr = 0.2, weight_decay = args.weight_decay)
    # change this to multistep after initial training
    # at each iteration the lr is multiplied by args.gamma (=0.7)
    #scheduler = ReduceLROnPlateau(optimizer, factor = 0.25, patience = 5)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # scheduler = CyclicLR(optimizer, base_lr = 0.02, max_lr = 0.4)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.2,
#                                                     steps_per_epoch=len(train_loader),
#                                                     epochs=args.epochs)

    Loss_monitor = pd.DataFrame(columns=["Train loss", "Test loss"])
    trainLossBatch = pd.DataFrame(columns=["epoch", "batchNum", "Train Loss"])
    print("Starting training :) \n\n")
    for epoch in range(1, args.epochs + 1):
        train_loss, lossPerBatch = train(args, net, device, train_loader, optimizer, epoch)
        test_loss = test(args, net, device, test_loader)

        Loss_monitor.loc[len(Loss_monitor)] = [train_loss, test_loss]
        trainLossBatch = trainLossBatch.append(lossPerBatch, ignore_index = True)
        # save results after each epoch (in case job does not finish)
        Loss_monitor.to_csv("Results/"+ args.resfile +"TrainTest.csv")
        trainLossBatch.to_csv("Results/"+args.resfile+"LossPBatch.csv")

        scheduler.step()
        #scheduler.step(test_loss) # if using ReduceLROnPlateau scheduler or another that requires this
        
    if args.save_net:
        torch.save(net.state_dict(), "HtoA.pt")


if __name__ == "__main__":
    main()
