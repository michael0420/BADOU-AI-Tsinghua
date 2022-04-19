import torch
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            use_bn=True,
            bn_eps=1e-5,
            activation='ReLU'
            ):
        super(ConvBlock,self).__init__()
        self.use_activ = (activation is not None)
        self.use_bn = use_bn
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias
            )
        if self.use_bn:
            self.bn = nn.BatchNorm2d(num_features=out_channels,eps=bn_eps)
        if self.use_activ:
            self.activation = nn.ReLU(inplace=True) if activation=='ReLU' else get_activation(activation)

    def forward(self,x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_activ:
            x = self.activation(x)
        return x
            

class ConvSeq(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channellist,
            kernel_sizelist,
            stridelist,
            paddinglist,
            convname
            ):
        super(ConvSeq,self).__init__()
        
        self.seq = nn.Sequential()
        for i in range(len(out_channellist)):
            self.seq.add_module(f'{convname}{i}',ConvBlock(
                in_channels=in_channels,
                out_channels=out_channellist[i],
                kernel_size=kernel_sizelist[i],
                stride=stridelist[i],
                padding=paddinglist[i])
                )
            in_channels = out_channellist[i]

    def forward(self,x):
        x = self.seq(x)

        return x


