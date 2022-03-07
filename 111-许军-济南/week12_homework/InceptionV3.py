# -- coding:utf-8 --
import torch.nn as nn
import torch
import torch.nn.functional as F

class InceptionV3(nn.Module):
    def __init__(self):
        super(InceptionV3,self).__init__()
        self.max_pool=nn.MaxPool2d(kernel_size=5, stride=1)
        self.dropout=nn.Dropout(p=0.5)
        self.linear=nn.Linear(2048, 2)

    def ConvBNRelu(self,inchannels,outchannels,kernel_size,stride,padding = 0):
        block1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels,out_channels=outchannels,kernel_size=kernel_size,stride = stride,padding=padding),
            nn.BatchNorm2d(outchannels),
            nn.ReLU6(inplace=True),
        )
        return block1

    def InceptionV3ModuleA(self,input_tensor,in_channels,out_channels,out_channels1,out_channels11,out_channels21,out_channels22,out_channels3):
        self.branch1 = self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels,kernel_size=1,stride=1)
        self.branch2 = nn.Sequential(
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels1,kernel_size=1,stride=1,padding=0),
            self.ConvBNRelu(inchannels=out_channels1,outchannels=out_channels11,kernel_size=5,stride=1,padding=2)
        )
        self.branch3 = nn.Sequential(
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels21,kernel_size=1,stride=1),
            self.ConvBNRelu(inchannels=out_channels21,outchannels=out_channels22,kernel_size=3,stride=1,padding=1),
            self.ConvBNRelu(inchannels=out_channels22, outchannels=out_channels22, kernel_size=3, stride=1,padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels3,kernel_size=1,stride=1)
        )
        out1 = self.branch1(input_tensor)
        out2 = self.branch2(input_tensor)
        out3 = self.branch3(input_tensor)
        out4 = self.branch4(input_tensor)
        out = torch.cat([out1,out2,out3,out4],dim=1)
        return out

    def InceptionV3ModuleB(self,input_tensor,in_channels,out_channels,out_channels1,out_channels11):
        self.branch1 = self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels,kernel_size=3,stride=2)

        self.branch2 = nn.Sequential(
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels1,kernel_size=1,stride=1),
            self.ConvBNRelu(inchannels=out_channels1,outchannels=out_channels11,kernel_size=3,stride=1,padding=1),
            self.ConvBNRelu(inchannels=out_channels11, outchannels=out_channels11, kernel_size=3, stride=2)
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        out1 = self.branch1(input_tensor)
        out2 = self.branch2(input_tensor)
        out3 = self.branch3(input_tensor)
        out = torch.cat([out1,out2,out3],dim=1)
        return out

    def InceptionV3ModuleC(self,input_tensor,in_channels,out_channels,out_channels1,out_channels11,out_channels21,out_channels22,out_channels3):
        self.branch1 = self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels,kernel_size=1,stride=1)
        self.branch2 = nn.Sequential(
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels1,kernel_size=1,stride=1),
            self.ConvBNRelu(inchannels=out_channels1,outchannels=out_channels1,kernel_size=[1,7],stride=1,padding=[0,3]),
            self.ConvBNRelu(inchannels=out_channels1, outchannels=out_channels11, kernel_size=[7, 1], stride=1,padding=[3, 0]))

        self.branch3 = nn.Sequential(
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels21,kernel_size=1,stride=1),
            self.ConvBNRelu(inchannels=out_channels21,outchannels=out_channels21,kernel_size=[7,1],stride=1,padding=[3,0]),
            self.ConvBNRelu(inchannels=out_channels21, outchannels=out_channels21, kernel_size=[1,7], stride=1,padding=[0,3]),
            self.ConvBNRelu(inchannels=out_channels21, outchannels=out_channels21, kernel_size=[7, 1], stride=1,padding=[3, 0]),
            self.ConvBNRelu(inchannels=out_channels21, outchannels=out_channels22, kernel_size=[1, 7], stride=1, padding=[0, 3]))

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels3,kernel_size=1,stride=1)
        )
        out1 = self.branch1(input_tensor)
        out2 = self.branch2(input_tensor)
        out3 = self.branch3(input_tensor)
        out4 = self.branch4(input_tensor)
        out = torch.cat([out1,out2,out3,out4],dim=1)
        return out

    def InceptionV3ModuleD(self,input_tensor,in_channels,out_channels,out_channels1,out_channels11):
        self.branch1 = nn.Sequential(
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels,kernel_size=1,stride=1),
            self.ConvBNRelu(inchannels=out_channels,outchannels=out_channels1,kernel_size=3,stride=2))

        self.branch2 = nn.Sequential(
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels11,kernel_size=1,stride=1),
            self.ConvBNRelu(inchannels=out_channels11,outchannels=out_channels11,kernel_size=[1,7],stride=1,padding=[0,3]),
            self.ConvBNRelu(inchannels=out_channels11, outchannels=out_channels11, kernel_size=[7,1], stride=1,padding=[3,0]),
            self.ConvBNRelu(inchannels=out_channels11,outchannels=out_channels11,kernel_size=3,stride=2)
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        out1 = self.branch1(input_tensor)
        out2 = self.branch2(input_tensor)
        out3 = self.branch3(input_tensor)
        out = torch.cat([out1,out2,out3],dim=1)
        return out

    def InceptionV3ModuleE(self,input_tensor,in_channels,out_channels,out_channels1,out_channels21,out_channels22,out_channels3):
        self.branch1 = self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels,kernel_size=1,stride=1)
        self.branch2 = nn.Sequential(self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels1,kernel_size=1,stride=1))
        s2 = self.branch2(input_tensor)
        self.branch21 = nn.Sequential(self.ConvBNRelu(inchannels=out_channels1,outchannels=out_channels1,kernel_size=[1,3],stride=1,padding=[0,1]))
        self.branch22 = nn.Sequential(self.ConvBNRelu(inchannels=out_channels1,outchannels=out_channels1,kernel_size=[3,1],stride=1,padding=[1,0]))
        s21 = self.branch21(s2)
        s22 = self.branch22(s2)
        s2_out = torch.cat([s21,s22],dim=1)
        self.branch3 = nn.Sequential(
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels21,kernel_size=1,stride=1),
            self.ConvBNRelu(inchannels=out_channels21,outchannels=out_channels22,kernel_size=3,stride=1,padding=1))
        s3 = self.branch3(input_tensor)
        self.branch31 = nn.Sequential(self.ConvBNRelu(inchannels=out_channels22,outchannels=out_channels22,kernel_size=[1,3],stride=1,padding=[0,1]))
        self.branch32 = nn.Sequential(self.ConvBNRelu(inchannels=out_channels22,outchannels=out_channels22,kernel_size=[3,1],stride=1,padding=[1,0]))
        s31 = self.branch31(s3)
        s32 = self.branch32(s3)
        s3_out = torch.cat([s31,s32],dim=1)
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
            self.ConvBNRelu(inchannels=in_channels,outchannels=out_channels3,kernel_size=1,stride=1)
        )
        out1 = self.branch1(input_tensor)
        out4 = self.branch4(input_tensor)
        out = torch.cat([out1,s2_out,s3_out,out4],dim=1)
        return out

    def forward(self,x):
        x = self.ConvBNRelu(3,32,3,2,0)(x)
        x = self.ConvBNRelu(32,32,3,1,0)(x)
        x = self.ConvBNRelu(32,64,3,1,0)(x)
        x = nn.MaxPool2d(kernel_size=3,stride=2)(x)
        x = self.ConvBNRelu(64,80,1,1,0)(x)
        x = self.ConvBNRelu(80,192,3,1,0)(x)
        x = nn.MaxPool2d(kernel_size=3,stride=2)(x)

        x = self.InceptionV3ModuleA(x,in_channels=192,out_channels=64,out_channels1=48,out_channels11=64,out_channels21=64,
                                   out_channels22=96,out_channels3=32 )
        x = self.InceptionV3ModuleA(x, in_channels=256, out_channels=64, out_channels1=48, out_channels11=64,
                                 out_channels21=64,out_channels22=96, out_channels3=64)
        x = self.InceptionV3ModuleA(x, in_channels=288, out_channels=64, out_channels1=48, out_channels11=64,
                                 out_channels21=64, out_channels22=96, out_channels3=64)
        x = self.InceptionV3ModuleB(x,in_channels=288,out_channels=384,out_channels1=64,out_channels11=96)
        x = self.InceptionV3ModuleC(x,in_channels=768,out_channels=192,out_channels1=128,out_channels11=192,out_channels21=128,out_channels22=192,
                                    out_channels3=192)
        x = self.InceptionV3ModuleC(x, in_channels=768, out_channels=192, out_channels1=160, out_channels11=192,
                                  out_channels21=160, out_channels22=192,out_channels3=192)
        x = self.InceptionV3ModuleC(x, in_channels=768, out_channels=192, out_channels1=160, out_channels11=192,
                                  out_channels21=160, out_channels22=192,out_channels3=192)
        x = self.InceptionV3ModuleC(x, in_channels=768, out_channels=192, out_channels1=192, out_channels11=192,
                                  out_channels21=192, out_channels22=192,out_channels3=192)
        x = self.InceptionV3ModuleD(x,in_channels=768,out_channels=192,out_channels1=320,out_channels11=192)
        x = self.InceptionV3ModuleE(x,in_channels=1280,out_channels=320,out_channels1=384,out_channels21=448,out_channels22=384,out_channels3=192)
        x = self.InceptionV3ModuleE(x, in_channels=2048, out_channels=320, out_channels1=384, out_channels21=448,out_channels22=384, out_channels3=192)
        print(x.size())
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.linear(x)
        return x