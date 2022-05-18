# -- coding:utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
class AlexNet(torch.nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=48,kernel_size=11,stride=4)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2 = nn.Conv2d(in_channels=48,out_channels=128,kernel_size=3,stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv3 = nn.Conv2d(in_channels=128,out_channels=196,kernel_size=3,stride=1)
        self.conv4 = nn.Conv2d(in_channels=196, out_channels=128, kernel_size=3, stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3,stride=2)
        self.fc1 = nn.Linear(3*3*128,500)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(500,2)
    def forward(self,x):
        o1 = F.relu(self.conv1(x))
        # o2 = nn.BatchNorm2d(o1)
        o3 = self.maxpool1(o1)
        o4 = F.relu(self.conv2(o3))
        o6 = self.maxpool2(o4)
        o7 = F.relu(self.conv3(o6))
        o8 = F.relu(self.conv4(o7))
        o9 = self.maxpool3(o8)
        o9 = o9.view(o9.size(0),-1)
        o10 = self.fc1(o9)
        o10 = self.dropout(o10)
        x = self.fc2(o10)
        x = F.softmax(x)
        print(x)
        return x

