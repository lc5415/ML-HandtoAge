import torch
from torchvision import transforms, utils
import pandas as pd
import os, re
import cv2
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import time
import torchvision.utils as utils
try: 
    from Scripts.MyTransforms import *
except:
    from MyTransforms import *
plt.rcParams['image.cmap'] = 'gray' # set default colormap to gray


class HandDataset(Dataset):
    """Hand labels dataset."""

    def __init__(self, pandas_df, data_labels, root_dir, transform=None, normalise = True, clahe = True, outputs = 1):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a image.
        """

        # pass dataframe containing all image labels (i.e. ids, e.g 1876,1746...)
        # of the images to be included in this loader
        self.labels_index = pandas_df  # pd.read_csv(csv_file)
        # pass data frame containing "target" i.e outcome/label for each image id
        self.labels_frame = data_labels
        # pass path of directory where images are locate
        self.root_dir = root_dir
        # pass whatever transform you'd like to apply to the data
        self.transform = transform
        self.normalise = normalise
        self.clahe = clahe
        self.outputs = 1 # number of outputs to spit

    def __len__(self):
        # return number of indices in this dataset (aka, number of images in whole dataset)
        return len(self.labels_index)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # get image ids
        image_id = self.labels_index.iloc[idx, 0]
        # make image path by preceding the image id by the root directory and
        # by appending .png after it
        img_name = os.path.join(self.root_dir,
                                image_id + ".png")
        # read image
        image = io.imread(img_name)
        # reshape image to [HxWx1]
        image = image.reshape((image.shape[0], image.shape[1], 1))

        ## get label along with image and make it into an array
        age = self.labels_frame.loc[self.labels_frame.id == int(image_id), 'boneage']
        age = np.array(age)

        #get sex as well. coded as male (1: male, 0: female)
        sex = self.labels_frame.loc[self.labels_frame.id == int(image_id), 'male']
        sex = np.array(sex)

        # put image and its labels in a sample dictionary
        sample = {'image': image, 'age': age, 'sex': sex}

        if self.transform:
            # transform with transform that was given to the function
            sample = self.transform(sample)

        # if self.outputs == 1:
        #     sample = {'image': image, 'age': age}
        image, age, sex = sample['image'], sample['age'], sample['sex']
        return image, age, sex # sample

    def plot_img(self, idx):

        """If transforms has been applied this will plot the transformed images"""

        # get sample with given id and retrieve the image from there
        img, _, _ = self.__getitem__(idx)

        #img = self.__getitem__(idx)['image'] # dictionary version
        # If a transform was applied reorder the image so that it can be plotted
        if self.transform:
            img = np.transpose(img, (1, 2, 0))
        # reshape image to  be HxW only,
        # explanation
        """matplotlib.pyplot does not like 2D tensors for grayscale images, if an
         image is grayscale plt likes it to be a simple matrix of dims HxW, for 
         RGB images it takes HxWxC images but grayscale images are not just HxWx1
         
         randimg = np.random.rand(256,256,3) # random RGB images
         --> no problem
         randimg  = np.random.rand(256,256) # grayscale image
         --> no problem
         randimg = np.random.rand(256,256,1) # grasycale img as 1D tensor
         --> error
         """
        img = img.reshape((img.shape[0], img.shape[1]))
        plt.imshow(img)
        plt.show()

    def plot_n_images(self, n_images=9):
        """
        This image will give you a matplotlib subplot of n images
        from a torch Dataset, given a bunch of indices

        Note: these will be transformed images if a transformation has been applied
        """

        # set seed to get same images this function is called
        np.random.seed(12435)
        fig = plt.figure()
        for k, img_id in enumerate(np.random.randint(0,
                                                     len(self.labels_index),
                                                     n_images)):

            # sample = self.__getitem__(img_id)
            # img = sample['image'] # dictionary version
            img, age, sex = self.__getitem__(img_id)

            # standard input size is [1,224,224] when transformed, as this is torch preferred
            # input format, line below is transposing to [224,224,1]
            if self.transform:
                img = np.transpose(img, (1, 2, 0))
            # this line is reshaping the tensor from [224,224,1] to [224,224]
            img = img.reshape((img.shape[0], img.shape[1]))
            ax = plt.subplot(3, 3, k + 1)
            # plh = np.zeros(n_images)
            # plh = plh.reshape([int(np.ceil(np.sqrt(n_images))),-1])
            # ax = plt.subplot(plh.shape[0], plh.shape[1], k+1)
            plt.imshow(img)
            plt.axis('off')
            # set title as age
            age_out = float((np.array(age) / 12))
            #ax.set_title(f"{age_out:.2f}, {'male' if sex == 1 else 'female'}")

        if img.shape[0] == img.shape[1]:
            all_axes = fig.get_axes()
            for ax in all_axes:
                ax.label_outer()
        #plt.tight_layout()
        plt.show()
        return fig

    def n_histograms(self, n_images=9):
        """
        This image will give you a matplotlib subplot of n images
        from a torch Dataset, given a bunch of indices

        Note: plots transformed images
        """

        # set seed to get same images this function is called
        np.random.seed(12435)
        fig = plt.figure()
        for k, img_id in enumerate(np.random.randint(0,
                                                     len(self.labels_index),
                                                     n_images)):
            # img = self.__getitem__(img_id)['image']
            img, _, _ = self.__getitem__(img_id)

            # standard input size is [1,224,224] when transformed, as this is torch preferred
            # input format, line below is transposing to [224,224,1]
            if self.transform:
                img = np.transpose(img, (1, 2, 0))
            # this line is reshaping the tensor from [224,224,1] to [224,224]
            img = img.reshape((img.shape[0], img.shape[1]))
            ax = plt.subplot(3, 3, k + 1)
            plt.grid()
            plt.hist(img.flatten(), edgecolor = 'black')
            try:
                img_mean = torch.mean(img)
                img_median = torch.median(img)
            except:
                img_mean = img.mean()
                img_median = np.median(img)
            plt.axvline(img_mean,
                        color='g',
                        linestyle='dashed',
                        linewidth=1)
            plt.axvline(img_median,
                        color='r',
                        linestyle='dashed',
                        linewidth=1)
            min_ylim, max_ylim = plt.ylim()
            min_xlim, max_xlim = plt.xlim()
            plt.text(0.5 * max_xlim, max_ylim * 0.8,
                     f"{img_mean - img_median:.2f}")
            plt.tight_layout()
            # ax.set_title(f'({img.shape[0]},{img.shape[1]})')

        all_axes = fig.get_axes()
        for ax in all_axes:
            ax.label_outer()
        plt.show()
        return fig

def Load(dataset, batch_size = 20, plot = 0):
    '''Given an object of the class torch.utils.data.Dataset this function
    returns a dataloader with the given images and the given labels and plots some stuff
    allong the way

    '''

    # detect number of cores, halve it and ceil it
    cores = int(np.ceil(multiprocessing.cpu_count()/2))

    if batch_size == "full":
        batch_size = dataset.labels_index.shape[0]

    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=0, drop_last = True)

    def show_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        if image.max() > 1:
            images_batch = image/255
        else:
            images_batch = image

        grid = utils.make_grid(images_batch, nrow= int(np.ceil(np.sqrt(images_batch.shape[0]))) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.show()
        grid2 = utils.make_grid(images_batch, nrow=int(np.ceil(np.sqrt(images_batch.shape[0]))))
        plt.hist(grid2.numpy().transpose((1, 2, 0)).flatten())
        plt.show()

    for i_batch, (image, age, sex) in enumerate(dataloader):

        # print(i_batch),
        #       sample_batched['image'].size(),
        #       sample_batched['age'].size()),
        #       sample_batched['sex'].size()) # dictionary version
        print(i_batch, image.size(), age.size(), sex.size())

        # observe 4th batch and stop.
        if i_batch == len(dataloader)-1 and plot != 0: # dataloader.batch_size - 1:
            plt.figure()
            show_batch(image)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

    return dataloader

def getData(image_directory, labels_directory, transform = None,
            plot = 0, batch_size = 20, save = 0, savename = ""):
    """Example: --getting all the data
    dataload = LoadImages.getData("labelled/train/",
...                           "boneage-training-dataset.csv",
...                           transform = transforms.Compose(
...                              [Rescale(256),
...                               RandomCrop(224),
...                               ToTensor()
...                               ]), batch_size = "full")

    --- getting mean and std values for whole batch:
    dataload = LoadImages.getData("labelled/train",
                                "boneage-training-dataset.csv",
                                transform = transforms.Compose(
                             [Rescale(256),
                               RandomCrop(224),
                              ToTensor()
                              ]), batch_size = "full", normalise = False)

    for id, batch in enumerate(data): print(batch['image'].mean(), batch['image'].std())

    You can also call FullBatchStatistics
    """
    # extract image names from shuffle of images I have obtained
    # training

    labels_indices = map(lambda filename: filename.split('.')[0], os.listdir(image_directory))
    labels_indices = pd.DataFrame(list(labels_indices))

    data_labels = pd.read_csv(labels_directory)

    DATASET = HandDataset(labels_indices,
                          data_labels,
                          image_directory,
                          transform= transform)

    if plot != 0:
        DATASET.plot_n_images()
        DATASET.n_histograms()

    Loader = Load(DATASET, batch_size = batch_size, plot = plot)

    if save:
        torch.save(Loader, os.getcwd()+"/labelled/"+savename)

    return Loader

def FullBatchStats(dataloaded):
    """This function expects an object of class torch.utils.data.DataLoader and
    will return the dataloader statistics (mean and std)"""

    if len(dataloaded) == 1:
        # for id, batch in enumerate(dataloaded):
        #     b_mean, b_std = batch['image'].mean(), batch['image'].std()
        for id, (image, _, _) in enumerate(dataloaded):
            b_mean, b_std = image.mean(), image.std()

    else:
        b_all, num_el = 0, 0
        # for id, batch in enumerate(dataloaded):
        #     b_all.append(batch['image'].flatten())
        for _, (image, _, _) in enumerate(dataloaded):
            b_all += image.flatten().sum()
            num_el += np.array(image.flatten().shape)
        b_mean = b_all/num_el
        b_all = 0
        for _, (image, _, _) in enumerate(dataloaded):
            b_all += ((image.flatten()-b_mean)**2).sum()
        b_std = (b_all/num_el).sqrt()

    return b_mean, b_std


# SAMPLE CODE
if __name__ == "__main__":
    st = time.time()
    data = getData("labelled/train",
                          "boneage-training-dataset.csv",
                          transform=transforms.Compose(
                                  [Rescale(256),
                                   # RandomCrop(224),
                                   CenterCrop(224),
                                   CHALE(),
                                   InstanceNorm(),
                                   ToTensor(),
                                   #Normalize([0.2011], [0.1847])
                                   ]), batch_size=20, plot = 1)
    # mean_full, std_full = FullBatchStats(data)
    print(time.time()-st)
