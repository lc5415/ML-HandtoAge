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
try: 
    from Scripts.MyTransforms import Rescale, RandomCrop, ToTensor, CenterCrop
except:
    from MyTransforms import Rescale, RandomCrop, ToTensor, CenterCrop
from torchvision.transforms import Normalize
plt.rcParams['image.cmap'] = 'gray' # set default colormap to gray


class HandDataset(Dataset):
    """Hand labels dataset."""

    def __init__(self, pandas_df, data_labels, root_dir, transform=None, normalise = True, outputs = 1):
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
        if self.normalise:
            """IN is hard coded at the moment which may not be ideal """
            # Instance Normalization
            image = sample['image']
            mean, std = torch.mean(image), torch.std(image)
            image = (image - mean) / std
            # return transformed image
            sample = {'image': image, 'age': age, 'sex': sex}

        # if self.outputs == 1:
        #     sample = {'image': image, 'age': age}

        return image, age, sex # sample

    def plot_img(self, idx):

        """If transforms has been applied this will plot the transformed images"""

        # get sample with given id and retrieve the image from there
        img, _ = self.__getitem__(idx)

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

    def plot_n_images(self, n_images=20):
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
            img, _ = self.__getitem__(img_id)

            # standard input size is [1,224,224] when transformed, as this is torch preferred
            # input format, line below is transposing to [224,224,1]
            if self.transform:
                img = np.transpose(img, (1, 2, 0))
            # this line is reshaping the tensor from [224,224,1] to [224,224]
            img = img.reshape((img.shape[0], img.shape[1]))
            ax = plt.subplot(4, 5, k + 1)
            plt.imshow(img)
            plt.tight_layout()
            # set title as age
            ax.set_title(f"{np.array(sample['age'])/12}")

        if img.shape[0] == img.shape[1]:
            all_axes = fig.get_axes()
            for ax in all_axes:
                ax.label_outer()
        plt.show()

    def n_histograms(self, n_images=20):
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
            img, _ = self.__getitem__(img_id)

            # standard input size is [1,224,224] when transformed, as this is torch preferred
            # input format, line below is transposing to [224,224,1]
            if self.transform:
                img = np.transpose(img, (1, 2, 0))
            # this line is reshaping the tensor from [224,224,1] to [224,224]
            img = img.reshape((img.shape[0], img.shape[1]))
            ax = plt.subplot(4, 5, k + 1)
            plt.hist(img.flatten())
            img_mean = torch.mean(img)
            img_median = torch.median(img)
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
                            shuffle=True, num_workers=cores, drop_last = True)

    def show_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch = sample_batched['image']

        grid = utils.make_grid(images_batch, nrow= int(np.ceil(np.sqrt(images_batch.shape[0]))) )
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
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
            show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

    return dataloader

def getData(image_directory, labels_directory, transform = None,
            normalise = True, plot = 0, batch_size = 20, save = 0, savename = ""):
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
                          transform= transform, normalise = normalise)

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
        b_all = []
        # for id, batch in enumerate(dataloaded):
        #     b_all.append(batch['image'].flatten())
        for id, (image, _, _) in enumerate(dataloaded):
            b_all.append(image.flatten())
        b_mean = b_all.mean()
        b_std = b_all.std()

    return b_mean, b_std


# SAMPLE CODE
# if __name__ == "__main__":
#     data = getData("labelled/train",
#                           "boneage-training-dataset.csv",
#                           transform=transforms.Compose(
#                                   [Rescale(256),
#                                    # RandomCrop(224),
#                                    CenterCrop(224),
#                                    ToTensor()
#                                    ]), batch_size="full", normalise=True, plot = 1)