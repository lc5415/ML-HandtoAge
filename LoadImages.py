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
from torch.nn.modules.normalization import GroupNorm
plt.rcParams['image.cmap'] = 'gray' # set default colormap to gray



def main():
    # extract image names from shuffle of images I have obtained
    # training
    training_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir("labelled/train/"))
    training_labels_indices = pd.DataFrame(list(training_labels_indices))

    class HandDataset(Dataset):
        """Hand labels dataset."""

        def __init__(self, pandas_df, root_dir, transform=None):
            """
            Args:
                csv_file (string): Path to the csv file with annotations.
                root_dir (string): Directory with all the images.
                transform (callable, optional): Optional transform to be applied
                    on a image.
            """
            self.labels_frame = pandas_df  # pd.read_csv(csv_file)
            self.root_dir = root_dir
            self.transform = transform

        def __len__(self):
            return len(self.labels_frame)

        def __getitem__(self, idx):
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_name = os.path.join(self.root_dir,
                                    self.labels_frame.iloc[idx, 0] + ".png")
            image = io.imread(img_name)
            image = image.reshape((image.shape[0], image.shape[1], 1))

            if self.transform:
                image = self.transform(image)

            return image

        def plot_img(self, idx):

            img = self.__getitem__(idx)
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
                                                         len(self.labels_frame),
                                                         n_images)):
                img = self.__getitem__(img_id)
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
                                                         len(self.labels_frame),
                                                         n_images)):
                img = self.__getitem__(img_id)
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

    class Rescale(object):
        """Rescale the image in a image to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """

        def __init__(self, output_size):
            # this assert checks whether output size is an int OR a tuple
            assert isinstance(output_size, (int, tuple))
            self.output_size = output_size

        def __call__(self, image):

            # line below equivalent to image.shape for 2D tensors
            h, w = image.shape[:2]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            resized_img = transform.resize(image, (new_h, new_w))

            return resized_img

    class RandomCrop(object):
        """Crop randomly the image in a image.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

        def __init__(self, output_size):
            assert isinstance(output_size, (int, tuple))
            if isinstance(output_size, int):
                self.output_size = (output_size, output_size)
            else:
                assert len(output_size) == 2
                self.output_size = output_size

        def __call__(self, image):

            h, w = image.shape[:2]
            new_h, new_w = self.output_size

            # getting new top of the image by choosing any random
            # number between 0 and height - new_height
            # must be: h > new_h and w > new_w
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            # new image is selected between the calculated top, which would
            # actually be bottom I feel and top +new_height
            image = image[top: top + new_h,
                    left: left + new_w]

            return image

    class ToTensor(object):
        """Convert ndarrays in image to Tensors."""

        def __call__(self, image):
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            return torch.from_numpy(image)

    trainDS = HandDataset(training_labels_indices, "labelled/train/",
                          transform=transforms.Compose(
                              [Rescale(256),
                               RandomCrop(224),
                               ToTensor()
                               ]))
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
        images_batch = sample_batched

        grid = utils.make_grid(images_batch, nrow=5)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))
        plt.show()

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched.size())

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
