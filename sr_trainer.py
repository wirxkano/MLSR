import copy 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from trainer.trainer import Trainer
from dataset.image_dataset import ImageDataset
from model.idn import IDN

from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

class SRTrainer(Trainer):
    def __init__(self, 
                 train_path,
                 val_path,
                 pretrain_path,
                 patch_size,
                 scale,
                 max_epoch, 
                 batch_size, 
                 pin_memory, 
                 have_validate=False, 
                 save_best_for=None, 
                 save_period=None, 
                 save_folder='./', 
                 snapshot_path=None, 
                 logger=None):
        
        self.train_path = train_path
        self.val_path = val_path
        self.pretrain_path = pretrain_path
        self.patch_size = patch_size
        self.scale = scale
        self.grad_num = 5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(max_epoch, 
                         batch_size,
                         pin_memory, 
                         have_validate, 
                         save_best_for, 
                         save_period, 
                         save_folder, 
                         snapshot_path, 
                         logger)
    
    # Get train dataset
    def build_train_dataset(self):
        return ImageDataset(self.train_path, patch_size=self.patch_size, scale=self.scale, train=True)
    
    # Get validate dataset
    def build_val_dataset(self):
        return ImageDataset(self.val_path, patch_size=self.patch_size, scale=self.scale, train=False)
    
    def build_test_dataset(self, test_path):
        return ImageDataset(test_path, patch_size=self.patch_size, scale=self.scale, train=False)
    
    # Get model
    def build_model(self):
        model = IDN(2, 64, 16, 4)
        checkpoint = torch.load(self.pretrain_path, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        return dict(
          final=model.to(self.device),
          update=copy.deepcopy(model).to(self.device),
          copy=copy.deepcopy(model).to(self.device)
        )
    
    # Get objective (loss) function
    def build_criterion(self):
        criterion = lambda x, y: F.l1_loss(x, y)
        return criterion
    
    # Get opimizer 
    def build_optimizer(self):
        return dict(
          final=optim.Adam(params=self.model["final"].parameters(), lr=0.0001),
          update=optim.SGD(params=self.model["update"].parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4),
          copy=optim.SGD(params=self.model["copy"].parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4),
        )

    # Get scheduler
    def build_scheduler(self):
        return dict(
          final=optim.lr_scheduler.MultiStepLR(self.optimizer["final"], [50, 100, 200], gamma=0.5)
        )
    
    # Design forward, backward and update process
    def train_step(self, batch):
        # Preprocess & Un-patch
        lr_son, lr_img, hr_img = batch
        lr_son, lr_img, hr_img = lr_son.to(self.device), lr_img.to(self.device), hr_img.to(self.device)
        # Load update parameters
        self.model["update"].load_state_dict(self.model["final"].state_dict())
        # Clear gradient from optimizer
        self.optimizer["final"].zero_grad()
        # Turn on gradient calculation flag
        with torch.set_grad_enabled(True):
            for _ in range(self.grad_num):
                # Forward
                self.optimizer["update"].zero_grad()
                out = self.model["update"](lr_son)
                # Loss Calculation
                loss = self.criterion(out, lr_img)
                # Calculate gradient (backward)
                loss.backward()
                # Update weights
                self.optimizer["update"].step()
            
            out = self.model["update"](lr_img)
            meta_loss = self.criterion(out, hr_img)
            meta_loss.backward()
            self.optimizer["final"].step()
            
        return dict(meta_loss=meta_loss.item())

    # Validate for each batch
    def validate_step(self, batch):
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        psnr_metric.reset()
        ssim_metric.reset()
        ## Preprocess & Un-patch
        lr_son, lr_img, hr_img = batch
        lr_son, lr_img, hr_img = lr_son.to(self.device), lr_img.to(self.device), hr_img.to(self.device)
        # Load update parameters
        self.model["copy"].load_state_dict(self.model["final"].state_dict())
        # Gradient update
        for _ in range(self.grad_num):
            self.model["copy"].train()
            self.optimizer["copy"].zero_grad()
            out = self.model["copy"](lr_son)
            loss = self.criterion(out, lr_img)
            loss.backward()
            self.optimizer["copy"].step()
            
        self.model["copy"].eval()
        with torch.no_grad():
            out = self.model["copy"](lr_img)
            psnr = psnr_metric(out, hr_img)
            ssim = ssim_metric(out, hr_img)
            
        return dict(pnsr=psnr.item(),
                    ssim=ssim.item())
