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
from MyTransforms import Rescale, RandomCrop, ToTensor
from torchvision.transforms import Normalize
from torch.nn.modules.normalization import GroupNorm
plt.rcParams['image.cmap'] = 'gray' # set default colormap to gray



def main():
    # extract image names from shuffle of images I have obtained
    # training
    training_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir("labelled/train/"))
    training_labels_indices = pd.DataFrame(list(training_labels_indices))
    data_labels = pd.read_csv("boneage-training-dataset.csv")

    class HandDataset(Dataset):
        """Hand labels dataset."""

        def __init__(self, pandas_df, data_labels,  root_dir, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a image.
            """
            self.labels_index = pandas_df  # pd.read_csv(csv_file)
            self.labels_frame = data_labels
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.labels_index)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            image_id = self.labels_index.iloc[idx, 0]
            img_name = os.path.join(self.root_dir,
                                     image_id + ".png")
            image = io.imread(img_name)
            image = image.reshape((image.shape[0], image.shape[1], 1))

            ## get label along with image
            age = self.labels_frame.loc[self.labels_frame.id == int(image_id), 'boneage']
            age = np.array(age)
            sample = {'image': image, 'age': age}

            if self.transform:
                sample = self.transform(sample)
                # Instance Normalization
                image = sample['image']
                mean, std = torch.mean(image), torch.std(image)
                image = (image - mean)/std

            sample = {'image': image, 'age': age}

            return sample

        def plot_img(self, idx):

            img = self.__getitem__(idx)['image']

            if self.transform:
                img = np.transpose(img, (1, 2, 0))
            img = img.reshape((img.shape[0], img.shape[1]))
            plt.imshow(img)
            plt.show()

        def plot_n_images(self, n_images=20):
            """
            This image will give you a matplotlib subplot of n images
            from a torch Dataset, given a bunch of indices
            """

            # set seed to get same images this function is called
            np.random.seed(12435)
            fig = plt.figure()
            for k, img_id in enumerate(np.random.randint(0,
                                                         len(self.labels_index),
                                                         n_images)):
                img = self.__getitem__(img_id)['image']

                # standard input size is [1,224,224] when transformed, as this is torch preferred
                # input format, line below is transposing to [224,224,1]
                if self.transform:
                    img = np.transpose(img, (1, 2, 0))
                # this line is reshaping the tensor from [224,224,1] to [224,224]
                img = img.reshape((img.shape[0], img.shape[1]))
                ax = plt.subplot(4, 5, k + 1)
                plt.imshow(img)
                plt.tight_layout()
                ax.set_title(f'{img_id}')

            if img.shape[0] == img.shape[1]:
                all_axes = fig.get_axes()
                for ax in all_axes:
                    ax.label_outer()
            plt.show()

        def n_histograms(self, n_images=20):
            """
            This image will give you a matplotlib subplot of n images
            from a torch Dataset, given a bunch of indices
            """

            # set seed to get same images this function is called
            np.random.seed(12435)
            fig = plt.figure()
            for k, img_id in enumerate(np.random.randint(0,
                                                         len(self.labels_index),
                                                         n_images)):
                img = self.__getitem__(img_id)['image']
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
                plt.text(0.5*max_xlim, max_ylim*0.8,
                         f"{img_mean-img_median:.2f}")
                plt.tight_layout()
                #ax.set_title(f'({img.shape[0]},{img.shape[1]})')

            all_axes = fig.get_axes()
            for ax in all_axes:
                ax.label_outer()
            plt.show()

    trainDS = HandDataset(training_labels_indices,
                          data_labels,
                          "labelled/train/",
                          transform=transforms.Compose(
                              [Rescale(256),
                               RandomCrop(224),
                               ToTensor()
                               ]))
                        # I'm not normalizing this way as I am normalizing on
                        # per image basis
                          # ,
                          #      transforms.Normalize(
                          #          mean=[0.406],
                          #          std=[0.225])]))

    trainDS.plot_n_images()
    trainDS.n_histograms()

    # detect number of cores
    cores = multiprocessing.cpu_count()

    dataloader = DataLoader(trainDS, batch_size=20,
                            shuffle=True, num_workers=cores)

    def show_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch = sample_batched['image']

        grid = utils.make_grid(images_batch, nrow=5)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.show()

    for i_batch, sample_batched in enumerate(dataloader):

        print(i_batch, sample_batched['image'].size(),
              sample_batched['age'].size())

        # observe 4th batch and stop.
        if i_batch == 3: # dataloader.batch_size - 1:
            plt.figure()
            show_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

    return dataloader

## image code to load image, set rescaler, scaled the img and plot it
# img = trainDS[2]
# scale = Rescale(256)
# scaled_img = scale(img)
# plt.imshow(scale(img))
# plt.show()

if __name__ == '__main__':
    main()
