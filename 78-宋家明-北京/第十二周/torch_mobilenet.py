import torch
from torch import nn
from common import ConvBlock

class MobileBlock(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            stride
            ):
        super(MobileBlock,self).__init__()

        self.conv1 = ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=stride,
                padding=1
                )
        self.conv2 = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0
                )

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class BigmobileBlock(nn.Module):
    def __init__(self):
        super(BigmobileBlock,self).__init__()
        
        self.convdw = MobileBlock(
                in_channels=512,
                out_channels=512,
                stride=1
                )
        self.conv = ConvBlock(
                in_channels=512,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0
                )
        self.seqnet = nn.Sequential()
        for i in range(5):
            self.seqnet.add_module(f'convdw{i*2+14}',self.convdw)
            self.seqnet.add_module(f'conv{i*2+15}',self.conv)

    def forward(self,x):
        x = self.seqnet(x)

        return x

class mobilenet(nn.Module):
    def __init__(self,class_num,in_size=(224,224)):
        super(mobilenet,self).__init__()
        self.in_size = in_size
        self.class_num = class_num

        self.conv1 = ConvBlock(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1
                )
        self.conv2dw = MobileBlock(
                in_channels=32,
                out_channels=32,
                stride=1
                )
        self.conv3 = ConvBlock(
                in_channels=32,
                out_channels=64,
                kernel_size=1,
                stride=1,
                padding=0
                )
        self.conv4dw = MobileBlock(
                in_channels=64,
                out_channels=64,
                stride=2
                )
        self.conv5 = ConvBlock(
                in_channels=64,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0
                )
        self.conv6dw = MobileBlock(
                in_channels=128,
                out_channels=128,
                stride=1
                )
        self.conv7 = ConvBlock(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0
                )
        self.conv8dw = MobileBlock(
                in_channels=128,
                out_channels=128,
                stride=2
                )
        self.conv9 = ConvBlock(
                in_channels=128,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0
                )
        self.conv10dw = MobileBlock(
                in_channels=256,
                out_channels=256,
                stride=1
                )
        self.conv11 = ConvBlock(
                in_channels=256,
                out_channels=256,
                kernel_size=1,
                stride=1,
                padding=0
                )
        self.conv12dw = MobileBlock(
                in_channels=256,
                out_channels=256,
                stride=2
                )
        self.conv13 = ConvBlock(
                in_channels=256,
                out_channels=512,
                kernel_size=1,
                stride=1,
                padding=0
                )
        self.BigmobileBlock = BigmobileBlock() 
        self.conv24dw = MobileBlock(
                in_channels=512,
                out_channels=512,
                stride=2
                )
        self.conv25 = ConvBlock(
                in_channels=512,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                padding=0
                )
        self.conv26dw = MobileBlock(
                in_channels=1024,
                out_channels=1024,
                stride=1
                )
        self.conv27 = ConvBlock(
                in_channels=1024,
                out_channels=1024,
                kernel_size=1,
                stride=1,
                padding=0
                )
        self.avgpool = nn.AvgPool2d(
                kernel_size=7,
                stride=1,
                padding=0
                )
        self.fc = nn.Linear(1024,self.class_num)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2dw(x)
        x = self.conv3(x)
        x = self.conv4dw(x)
        x = self.conv5(x)
        x = self.conv6dw(x)
        x = self.conv7(x)
        x = self.conv8dw(x)
        x = self.conv9(x)
        x = self.conv10dw(x)
        x = self.conv11(x)
        x = self.conv12dw(x)
        x = self.conv13(x)
        x = self.BigmobileBlock(x)
        x = self.conv24dw(x)
        x = self.conv25(x)
        x = self.conv26dw(x)
        x = self.conv27(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = self.softmax(x)
        return x

def main_test(device,batch_size,class_num):

    x = torch.randn((batch_size,3,224,224),device=device)

    model = mobilenet(class_num).to(device)

    model.eval()

    pred = model(x)
    print(pred.shape)
    print(pred.argmax(0))

if __name__=='__main__':
    """
    torch 实现mobilenet v1
    """

    batch_size = 8
    class_num = 10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main_test(device,batch_size,class_num)

        



