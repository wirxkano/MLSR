from dataset import ImageDataset
from torch.utils.data import DataLoader
from const import *
from train import SRTrainer
from model import IDNModel

def main():
  train_dataset = ImageDataset(train_hr_dir, train_lr_bicubic_dir, patch_size=128, train=True)
  val_dataset = ImageDataset(val_hr_dir, val_lr_bicubic_dir, patch_size=128, train=False)
  
  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

  trainer = SRTrainer(train_loader, val_loader, IDNModel)
  trainer.train()
    
if __name__ == '__main__':
  main()
