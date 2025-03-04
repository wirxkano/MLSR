import torch
import torch.nn as nn
import torch.nn.functional as F
from IDN import IDN
from skimage.metrics import structural_similarity as ssim
import cv2

class IDNModel(nn.Module):
    def __init__(self):
        super(IDNModel, self).__init__()
        self.idn = IDN()
    
    def forward(self, img_lr, img_bicubic, gt_output):
        output = self.idn(img_lr, img_bicubic)
        loss = F.mse_loss(output, gt_output)
        
        output_y = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
        gt_output_y = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
        img_bicubic_y = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)
        
        output = torch.clamp(output, 0, 255)
        
        psnr = self.psnr(output, gt_output)
        psnr_y = self.psnr(output_y, gt_output_y)
        ssim = self.ssim(output, gt_output)
        ssim_y = self.ssim(output_y, gt_output_y)
        bicubic_psnr = self.psnr(img_bicubic_y, gt_output_y)
        
        return {
            'output': output,
            'loss': loss,
            'psnr': psnr,
            'psnr_y': psnr_y,
            'ssim': ssim,
            'ssim_y': ssim_y,
            'bicubic_psnr': bicubic_psnr
        }
        
    def psnr(self, output, gt_output):
        mse = F.mse_loss(output, gt_output)
        return 10 * torch.log10(255.0 ** 2 / mse)
    
    def ssim(self, output, gt_output):
        return ssim(output, gt_output, channel_axis=2)
