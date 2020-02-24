import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms # maybe will use this in the future
from LoadImages import getData
from MyTransforms import Rescale, RandomCrop, ToTensor
from MyResNet import ResNet, BasicBlock, Bottleneck
import utils
import numpy as np

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        # get the inputs in correct format and pass them onto CPU/GPU
        data, target = batch['image'].to(device), batch['age'].to(device)

        data = data.double()
        target = target.double()
        # initialize the parameters at 0
        optimizer.zero_grad()

        output = model(data)
        # compute loss
        ## change code below from -- we don't wanna be using NLL loss for regression
        # -- loss = F.nll_loss(output, target)
        ## to
        loss = F.mse_loss(output, target)
        train_loss += torch.sum(loss)
        loss.backward()
        optimizer.step()
        
        # the train epoch message will only be printed when the batch_id module the log_interval
        # argument is equal to 0 (log interval = 10). If the number of batches is lower than 10, 
        # it will never print anything
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * train_loader.batch_sampler.batch_size,
                len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))

    return train_loss/len(train_loader)


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch['image'].to(device), batch['age'].to(device)
            data = data.double()
            target = target.double()
            output = model(data)
            test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Hand-to-Age ML Project')


    ## my code does not really take this into account
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    # nor does it takes this into account
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')

    ## will likely need to change this
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 14)')

    ## may need to do CV for this
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')

    ## may need to do CV for this
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=8, metavar='N',
                        help='how many batches to wait before logging training status')
    ## may want to do this when submitting multi-node jobs for cross-val
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print("You are using a "+str(device))

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    ### LOAD DATA
    train_loader = getData("../labelled/train/", "../boneage-training-dataset.csv",
                           transform = transforms.Compose(
                              [Rescale(256),
                               RandomCrop(224),
                               ToTensor()
                               ]),
                           plot = 0, batch_size = 16)

    test_loader = getData("../labelled/test/", "../boneage-training-dataset.csv",
                           transform=transforms.Compose(
                               [Rescale(256),
                                RandomCrop(224),
                                ToTensor()
                                ]),
                           plot=0, batch_size="full")


    architectures = [(BasicBlock, [2, 2, 2, 2]),
                     (BasicBlock, [3, 4, 6, 3]),
                     (Bottleneck, [3, 4, 6, 3]),
                     (Bottleneck, [3, 4, 23, 3])]

    # may need to change this, need to read on blocks and layers OR ask Arinbjorn
    net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=1)
    net = net.double()
    model = net.to(device)

    # does the optimizer I use matter?
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)


    # what is this?
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(args, model, device, test_loader)

        # plot loss to visdom object
        plotter.plot('loss', 'train', 'Class Loss', epoch, np.array(torch.mean(train_loss)))

        # plot loss to visdom object
        plotter.plot('loss', 'test', 'Class Loss', epoch, np.array(torch.mean(test_loss)))


        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "HtoA.pt")


if __name__ == "__main__":
    global plotter
    plotter = utils.VisdomLinePlotter(env_name = 'Training curves')
    main()
