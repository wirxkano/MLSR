import torch
import torch.nn as nn
import torch.nn.functional as F

class LReLU(nn.Module):
    def __init__(self, alpha=0.05):
        super(LReLU, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return torch.maximum(self.alpha * x, x)

class GroupConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=4, padding=1, activation=True):
        super(GroupConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True)
        self.activation = LReLU() if activation else None
    
    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        return x

class DistillationBlock(nn.Module):
    def __init__(self):
        super(DistillationBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 48, kernel_size=3, padding=1)
        self.conv2 = GroupConv2d(48, 48)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(48, 64, kernel_size=3, padding=1)
        self.conv5 = GroupConv2d(64, 48)
        self.conv6 = nn.Conv2d(48, 80, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(80 + 64, 64, kernel_size=1)
        self.activation = LReLU()

    def forward(self, x):
        tmp = self.activation(self.conv1(x))
        tmp = self.activation(self.conv2(tmp))
        tmp = self.activation(self.conv3(tmp))
        tmp1, tmp2 = torch.split(tmp, [16, 48], dim=1)
        tmp2 = self.activation(self.conv4(tmp2))
        tmp2 = self.activation(self.conv5(tmp2))
        tmp2 = self.activation(self.conv6(tmp2))
        output = torch.cat([x, tmp1], dim=1) + tmp2
        output = self.activation(self.conv7(output))
        return output

class Upsample(nn.Module):
    def __init__(self, scale, features):
        super(Upsample, self).__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, 3 * (scale ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pixel_shuffle(x)
        return x

class IDN(nn.Module):
    def __init__(self, scale=2):
        super(IDN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.distill_blocks = nn.Sequential(*[DistillationBlock() for _ in range(4)])
        self.upsample = Upsample(scale, features=64)
    
    def forward(self, img_lr, img_bicubic):
        x = F.leaky_relu(self.conv1(img_lr), negative_slope=0.05)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.05)
        x = self.distill_blocks(x)
        x = self.upsample(x)
        return x + img_bicubic
