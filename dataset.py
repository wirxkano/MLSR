# ref: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import random

class ImageDataset(Dataset):
    def __init__(self, directory, patch_size, train=True):
        self.directory = directory
        self.image_list = os.listdir(directory)
        assert len(self.image_list) > 0, "Empty dataset!"
        self.patch_size = patch_size # cut image into patch
        self.train = train

    def augmentation(self, image):
        aug_methods = [
            Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
            Image.ROTATE_90, Image.ROTATE_180, 
            Image.ROTATE_270, Image.TRANSPOSE,
            Image.TRANSVERSE
        ]
        if np.random.randint(len(aug_methods) + 1) != 0:
            return image.transpose(random.choice(aug_methods))
        return image

    def __getitem__(self, index):
        img_path = os.path.join(self.directory, self.image_list[index])
        image = Image.open(img_path).convert('RGB')
        image = image.crop((0, 0, image.width - image.width % 8, image.height - image.height % 8))

        # if train -> crop image randomly
        if self.train:
            left = np.random.randint(image.width - self.patch_size)
            upper = np.random.randint(image.height - self.patch_size)
            image = image.crop((left, upper, left + self.patch_size, upper + self.patch_size))
        
        image = self.augmentation(image)

        img_hr_shape = image.size
        img_lr_shape = (img_hr_shape[0] // 2, img_hr_shape[1] // 2)
        img_downscale_shape = (img_lr_shape[0] // 2, img_lr_shape[1] // 2)

        img_hr = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        img_lr = np.array(image.resize(img_lr_shape, Image.BICUBIC), dtype=np.float32).transpose(2, 0, 1) / 255.0
        img_bicubic = np.array(image.resize(img_hr_shape, Image.BICUBIC), dtype=np.float32).transpose(2, 0, 1) / 255.0

        return torch.tensor(img_lr), torch.tensor(img_bicubic), torch.tensor(img_hr)

    def __len__(self):
        return len(self.image_list)
