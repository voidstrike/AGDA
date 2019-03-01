import gzip
import os
import pickle
import urllib
import h5py

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms

class USPS16(data.Dataset):

    def __init__(self, h5_path, train=True, transform=None):
        hf = h5py.File(h5_path, 'r')
        if train:
            self.dataframe = hf.get('train')
        else:
            self.dataframe = hf.get('test')

        self.feature_matrix = self.dataframe.get('data')[:]
        # self.feature_matrix *= 255.0
        self.feature_matrix = self.feature_matrix.reshape(-1, 1, 16, 16)
        self.feature_matrix = self.feature_matrix.transpose(
             (0, 2, 3, 1))

        label_list = self.dataframe.get('target')[:]
        self.label_list = label_list.reshape(1, -1)

        self.transform = transform

        hf.close()

    def __len__(self):
        return len(self.feature_matrix)

    def __getitem__(self, index):
        img, label = self.feature_matrix[index, ::], self.label_list[0, index]
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([np.int64(label).item()])
        # label = torch.FloatTensor([label.item()])
        return img, label

def get_usps(path, train):
    """Get USPS dataset loader."""
    # image pre-processing
    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(
                                          [.5, .5, .5],
                                          [.5, .5, .5])])

    # dataset and data loader
    usps_dataset = USPS16(path,
                        train=train,
                        transform=pre_process)

    usps_data_loader = torch.utils.data.DataLoader(
        dataset=usps_dataset,
        batch_size=128,
        shuffle=True)

    return usps_data_loader