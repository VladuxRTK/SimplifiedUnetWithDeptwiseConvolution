import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super(DepthwiseSeparableConv,self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size = kernel_size,
                                   padding=padding, stride=stride, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ResidualDepthwiseSeParableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualDepthwiseSeParableBlock,self).__init__()
        self.depthwise_separable_conv = DepthwiseSeparableConv(in_channels, out_channels,3, stride=stride)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BacthNorm2d(out_channels)

            )

    def forward(self, x):
        out = self.depthwise_separable_conv(x)
        out += self.shortcut(x)
        return F.relu(out)
