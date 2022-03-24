# -- coding:utf-8 --
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict

class BaseBlock(nn.Module):
    def __init__(self,inplanes,planes):
        super(BaseBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=1,stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(planes,planes*2,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out += residual
        return out

class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet,self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3,self.inplanes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)
        self.layer1 = self._make_layer(64,1)
        self.layer2 = self._make_layer(128,2)
        self.layer3 = self._make_layer(256,8)
        self.layer4 = self._make_layer(512,8)
        self.layer5 = self._make_layer(1024,4)
        # 进行权值初始化
        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                n=m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) :
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 =self.layer4(out3)
        out5 = self.layer5(out4)
        return out3,out4.out5


    def _make_layer(self,inplane,block):
        layer = []
        layer.append(("ds_conv",nn.Conv2d(int(inplane/2),inplane,kernel_size=3,stride=2,padding=1,bias=False)))
        layer.append(("ds_bn",nn.BatchNorm2d(inplane)))
        layer.append(("ds_relu",nn.LeakyReLU(0.1)))
        for i in range(block):# 注意
            layer.append(("residual_{}".format(i),BaseBlock(inplane,int(inplane/2))))
        return nn.Sequential(OrderedDict(layer))
