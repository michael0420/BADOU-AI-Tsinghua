# -- coding:utf-8 --
import torch
import  torch.nn  as nn
import torch.nn.functional as F

class ResNet(torch.nn.Module):
    def __init__(self):
        super(ResNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )



    def identitiy_block(self,input_tensor,in_place,place,stride = 1,expansion = 4):
       bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_place,out_channels=place,kernel_size=1),
            nn.BatchNorm2d(place),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=place,out_channels=place,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(place),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels=place,out_channels=place*expansion,kernel_size=1,stride=1),
            nn.BatchNorm2d(place*expansion)
        )
       return bottleneck(input_tensor) + input_tensor

    def cov_block(self,input_tensor,in_place,place,stride = 1,expansion = 4):
        bottleneck=nn.Sequential(
            nn.Conv2d(in_channels=in_place, out_channels=place, kernel_size=1),
            nn.BatchNorm2d(place),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=place, out_channels=place, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(place),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=place, out_channels=place * expansion, kernel_size=1, stride=1),
            nn.BatchNorm2d(place * expansion)
        )

        residual_block = nn.Sequential(
            nn.Conv2d(in_channels=in_place,out_channels=place*expansion,kernel_size=1,stride=stride),
            nn.BatchNorm2d(place*expansion),
        )
        return bottleneck(input_tensor)+residual_block(input_tensor)

    def forward(self,x):
        x = self.conv1(x)
        x = self.cov_block(x,64,64,stride=1,expansion=4)
        x = self.identitiy_block(x,256,64,stride=1,expansion=4)
        x = self.identitiy_block(x, 256, 64, stride=1, expansion=4)
        print(x)

        x = self.cov_block(x,256,128,stride=2,expansion=4)
        x = self.identitiy_block(x,512,128,stride=1,expansion=4)
        x = self.identitiy_block(x, 512, 128, stride=1, expansion=4)
        x = self.identitiy_block(x, 512, 128, stride=1, expansion=4)

        x = self.cov_block(x, 512, 256, stride=2, expansion=4)
        x = self.identitiy_block(x, 1024, 256, stride=1, expansion=4)
        x = self.identitiy_block(x, 1024, 256, stride=1, expansion=4)
        x = self.identitiy_block(x, 1024, 256, stride=1, expansion=4)
        x = self.identitiy_block(x, 1024, 256, stride=1, expansion=4)
        x = self.identitiy_block(x, 1024, 256, stride=1, expansion=4)

        x = self.cov_block(x,1024,512,stride=2,expansion=4)
        x = self.identitiy_block(x,2048,512,stride=1,expansion=4)
        x = self.identitiy_block(x,2048,512, stride=1, expansion=4)

        x = nn.AvgPool2d(kernel_size=7)(x)
        x = x.view(x.size(0),-1)
        x = nn.Linear(2048,2)(x)
        x = F.softmax(x)
        return x
