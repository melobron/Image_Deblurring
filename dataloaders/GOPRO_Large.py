from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os
from glob import glob
import cv2


class GOPRO_Large_train_64(Dataset):
    def __init__(self, blur_gamma=False):
        super(GOPRO_Large_train_64, self).__init__()

        self.sharp_path = './datasets/GOPRO_Large/train_64/**/sharp/'
        self.blur_path = './datasets/GOPRO_Large/train_64/**/blur/'
        self.blur_gamma_path = './datasets/GOPRO_Large/train_64/**/blur_gamma/'

        self.sharp_files = sorted(glob(os.path.join(self.sharp_path, '*.png')))
        if blur_gamma:
            self.blur_files = sorted(glob(os.path.join(self.blur_gamma_path, '*.png')))
        else:
            self.blur_files = sorted(glob(os.path.join(self.blur_path, '*.png')))

        self.sharp_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.blur_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        sharp_numpy = cv2.cvtColor(cv2.imread(self.sharp_files[index]), cv2.COLOR_BGR2RGB)
        blur_numpy = cv2.cvtColor(cv2.imread(self.blur_files[index]), cv2.COLOR_BGR2RGB)
        sharp_torch = self.sharp_transform(sharp_numpy)
        blur_torch = self.blur_transform(blur_numpy)
        return sharp_torch, blur_torch

    def __len__(self):
        return len(self.sharp_files)


class GOPRO_Large_test(Dataset):
    def __init__(self, blur_gamma=False):
        super(GOPRO_Large_test, self).__init__()

        self.sharp_path = './datasets/GOPRO_Large/test_64/**/sharp/'
        self.blur_path = './datasets/GOPRO_Large/test_64/**/blur/'
        self.blur_gamma_path = './datasets/GOPRO_Large/test/**/blur_gamma/'

        self.sharp_files = sorted(glob(os.path.join(self.sharp_path, '*.png')))
        if blur_gamma:
            self.blur_files = sorted(glob(os.path.join(self.blur_gamma_path, '*.png')))
        else:
            self.blur_files = sorted(glob(os.path.join(self.blur_path, '*.png')))

        self.sharp_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.blur_transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        sharp_numpy = cv2.cvtColor(cv2.imread(self.sharp_files[index]), cv2.COLOR_BGR2RGB)
        blur_numpy = cv2.cvtColor(cv2.imread(self.blur_files[index]), cv2.COLOR_BGR2RGB)
        sharp_torch = self.sharp_transform(sharp_numpy)
        blur_torch = self.blur_transform(blur_numpy)
        return sharp_torch, blur_torch

    def __len__(self):
        return len(self.sharp_files)
