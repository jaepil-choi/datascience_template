import torch
from torch.utils.data import Dataset

from torchvision import transforms
from albumentations import A

import pandas as pd
import numpy as np

from tqdm import tqdm

class CustomAugmentation:
    def __init__(self, resize, mean, std, **args):
        self.transform = transforms.Compose([
            transforms.CenterCrop((320, 256))
        ])

    def __call__(self, image):
        return self.transform(image)

class MyDataset(Dataset):
    """Some Information about MyDataset"""
    def __init__(self, csv_path, transform=None, mean=None, std=None, val_ratio=0.2):
        super(MyDataset, self).__init__()
        self._csv_path = csv_path
        self._transform = transform
        
        self.mean = mean
        self.std = std

        self.val_ratio = val_ratio

        self.setup()
        self.calc_statistics()
    
    def setup(self):
        self.df = pd.read_csv(self._csv_path)
        ## Add more code ##

    def calc_statistics(self):
        has_stats = self.mean is not None and self.std is not None
        if not has_stats:
            print('Calculating statistics')

            ## Add more code ##
            self.mean = None
            self.std = None

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        row = self.df.iloc[index, :] # Change ":" to column index if needed. 
        
        ## Add more code ##
        d1 = row['col1']
        d2 = row['col2']

        data = {'d1': d1, 'd2': d2}

        if self._transform:
            data = self._transform(data)
        
        return data

    def __len__(self):
        return len(self.df)
    
    def split_dataset(self):
        n_val = int(len(self) * self.val_ratio)
        n_train = len(self) - n_val
        train_set, val_set = torch.utils.data.random_split(self, [n_train, n_val])

        return train_set, val_set

    @staticmethod
    def denormalize(image, mean, std):
        ## Add more code ##
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)

        return img_cp
    