from skimage import transform
import numpy as np
import torch
from torchvision.transforms import Normalize as tchNormalize
import cv2

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

    def __call__(self, sample):

        image = sample.get('image')
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

        return {'image': resized_img, 'age': sample['age'], 'sex': sample['sex']}


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

    def __call__(self, sample):

        image = sample['image']
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

        return {'image': image, 'age': sample['age'], 'sex': sample['sex']}

class CenterCrop(object):
    """Crop at the center

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

        image = sample['image']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        h_center, w_center = int(np.ceil(h/2)), int(np.ceil(w/2))

        image = image[int(h_center-new_h/2): int(h_center + new_h/2),
                int(w_center-new_w/2): int(w_center + new_w/2)]

        return {'image': image, 'age': sample['age'], 'sex': sample['sex']}


class ToTensor(object):
    """Convert ndarrays in image to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = sample['image']
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).double(),
                'age': torch.from_numpy(sample['age']),
                'sex': torch.from_numpy(sample['sex'])}

class Normalize(object):
    """Just adapting the normalise function to my case"""

    def __init__(self, mean_in, std_in):
        self.mean = mean_in
        self.std = std_in
        self.Normal = tchNormalize(mean_in, std_in)

    def __call__(self, sample):
        image = sample['image']
        norm_im = self.Normal(tensor=image)
        return {'image': norm_im, 'age': sample['age'], 'sex': sample['sex']}

class CHALE(object):

    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = clipLimit
        self.tileGridSize = tileGridSize
        self.CHALE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, sample):
        image = sample['image']
        image = np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX))
        image = self.CHALE.apply(image)
        image = image[:, :, np.newaxis]
        #image = image.double()
        # return transformed image
        return {'image': image, 'age': sample['age'], 'sex': sample['sex']}

class InstanceNorm(object):

    def __init__(self, kind = 'meannorm'):
        """kind possible inputs are meannorm for mean normalisation or minmax
        or standard for standardisation"""
        self.kind = 'meannorm'
    def __call__(self, sample):
        # Instance Normalization
        image = sample['image']
        if self.kind == "meannorm":
            image = (image - image.mean()) / (image.max() - image.min())
        elif self.kind == "minmax":
            image = (image - image.min()) / (image.max() - image.min())
        elif self.kind == "standard":
            mean, std = torch.mean(image), torch.std(image)
            image = (image - mean) / std
        else:
            print("Sorry wrong input, please check the documentation")
            raise
        # return transformed image
        return {'image': image, 'age': sample['age'], 'sex': sample['sex']}
