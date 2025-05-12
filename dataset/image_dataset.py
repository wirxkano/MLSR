import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, hr_dir, patch_size=96, scale=4, train=True):
        self.hr_dir = hr_dir
        self.image_files = sorted(os.listdir(self.hr_dir))
        self.image_length = len(self.image_files)
        self.scale = scale
        self.lr_scale = self.scale // 2
        self.patch_size = patch_size
        self.train = train
        self.transform = self._build_transform()
        
    def _build_transform(self):
        transform = transforms.Compose([
            transforms.RandomCrop(self.patch_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((0, 90)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # to [-1, 1]
        ])
        
        return transform

    def __getitem__(self, index):
        hr_path = os.path.join(self.hr_dir, self.image_files[index])
        hr_img = Image.open(hr_path).convert('RGB')
        
        if self.train:
            hr_img = self.transform(hr_img)
        
        hr_img = hr_img.crop((0, 0, hr_img.width - hr_img.width % (self.scale*2), hr_img.height - hr_img.height % (self.scale*2)))
        lr_img = hr_img.resize((hr_img.width // self.scale, hr_img.height // self.scale), Image.BICUBIC)
        lr_son = lr_img.resize((lr_img.width // self.scale, lr_img.height // self.scale), Image.BICUBIC)
        
        hr_img = transforms.ToTensor()(hr_img)
        lr_img = transforms.ToTensor()(lr_img)
        lr_son = transforms.ToTensor()(lr_son)
        
        return lr_son, lr_img, hr_img

    def __len__(self):
        return self.image_length
    