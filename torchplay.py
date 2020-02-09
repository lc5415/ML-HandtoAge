import torch
from torchvision import transforms, utils
import pandas as pd
import os, re
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['image.cmap'] = 'gray' # set default colormap to gray


# extract image names from shuffle of images I have obtained
#training
training_labels_indices = map(lambda filename: filename.split('.')[0], os.listdir("toy_training/"))
training_labels_indices = pd.DataFrame(list(training_labels_indices))

class HandDataset(Dataset):
    """Hand labels dataset."""

    def __init__(self, pandas_df, root_dir, transform = None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.labels_frame = pandas_df # pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels_frame.iloc[idx, 0]+".png")
        image = io.imread(img_name)
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample

    def plot_img(self, idx, rescale = 0, multiple_images = 1):

        img = self.__getitem__(idx)
        if rescale == 1:
            scale = Rescale((256, 256))
            img = scale(img)
        plt.imshow(img)
        plt.show()

    def plot_n_images(self, n_images=12, rescale=0):
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
            if rescale == 1:
                scale = Rescale(256)
                img = scale(img)
            ax = plt.subplot(3, 4, k + 1)
            plt.imshow(img)
            plt.tight_layout()
            ax.set_title(f'{img.shape}')
        plt.show()


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        # this assert checks whether output size is an int OR a tuple
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

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

## RandomCrop can be used for data augmentation, not interested in this for now
class RandomCrop(object):
    """Crop randomly the image in a sample.

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

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


trainDS = HandDataset(training_labels_indices, "toy_training/")

# images can now be called by precising
# trainDS[2] for the second image or
# for image in training_labels_indices: bla bla bla

trainDS.plot_n_images()

## Sample code to load image, set rescaler, scaled the img and plot it
# img = trainDS[2]
# scale = Rescale(256)
# scaled_img = scale(img)
# plt.imshow(scale(img))
# plt.show()

trainDS.plot_n_images(rescale = 1)