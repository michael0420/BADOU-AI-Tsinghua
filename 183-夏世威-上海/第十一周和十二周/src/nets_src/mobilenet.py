import torch
from torch import nn
import torch.nn.functional as F


class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(DepthWiseConv, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=3, groups=in_channel, stride=stride)  # depthwise卷积
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride)  # pointwise卷积
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.dw_conv1 = DepthWiseConv(32, 64, stride=2)
        self.dw_conv2 = DepthWiseConv(64, 128, stride=2)
        self.dw_conv3 = DepthWiseConv(128, 256)
        self.dw_conv4 = DepthWiseConv(256, 512)

        self.avgpool = nn.MaxPool2d(kernel_size=3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dw_conv1(x)
        x = self.dw_conv2(x)
        x = self.dw_conv3(x)
        x = self.dw_conv4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = nn.Linear(512, 2)(x)
        return x
