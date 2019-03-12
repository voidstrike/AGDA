from torchvision.datasets import SVHN
from PIL import Image
import numpy as np


class GSVHN(SVHN):
    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False):
        super(GSVHN, self).__init__(
            root, split, transform, target_transform, download)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        # Convert to grayscale
        img = img.convert('L')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target