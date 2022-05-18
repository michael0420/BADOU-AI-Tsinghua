# -- coding:utf-8 --
import  torch.nn as nn
import torch
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet,self).__init__()
        self.cfg = [(32,64,1),(64,128,2),(128,128,1),(128,256,2),(256,256,1),(256,512,2),(512,512,1),(512,512,1),(512,512,1),(512,512,1),(512,512,1),(512,1024,2),(1024,1024,1)]
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=2,padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.avg = nn.AvgPool2d(kernel_size=7,stride=1)
        self.fc = nn.Linear(1024,2)


    def conv_dw(self,in_channels,out_channels,stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride = stride,padding=1,groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)

        )

    def make_layer(self):
        layers = []
        for parameters in self.cfg:
            layers.append(self.conv_dw(parameters[0],parameters[1],parameters[2]))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.make_layer()(x)
        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = nn.Softmax()(x)
        return x



