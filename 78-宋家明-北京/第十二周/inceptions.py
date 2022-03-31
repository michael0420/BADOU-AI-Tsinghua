import torch
from torch import nn
from common import ConvBlock, ConvSeq

def get_reducta():
    channels = [[384],[64,96,96]]
    kernel_sizes = [[3],[1,3,3]]
    strides = [[2],[1,1,2]]
    paddings = [[0],[0,1,0]]

    return channels,kernel_sizes,strides,paddings


class inceptionv3_a(nn.Module):
    def __init__(self,
            in_channels,
            out_channels
            ):
        super(inceptionv3_a,self).__init__()

        pool_channels = out_channels - 224
        self.branch1 = ConvBlock(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=1,
                stride=1
                )
        self.branch2 = nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=48,
                    kernel_size=1,
                    stride=1
                    ),
                ConvBlock(
                    in_channels=48,
                    out_channels=64,
                    kernel_size=5,
                    stride=1,
                    padding=2
                    )
                )
        self.branch3 = nn.Sequential(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=64,
                    kernel_size=1,
                    stride=1
                    ),
                ConvBlock(
                    in_channels=64,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1
                    ),
                ConvBlock(
                    in_channels=96,
                    out_channels=96,
                    kernel_size=3,
                    stride=1,
                    padding=1
                    )
                )
        self.branch4 = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=3,
                    stride=1,
                    padding=1
                    ),
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=pool_channels,
                    kernel_size=1,
                    stride=1
                    )
                )

    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        x = torch.cat([branch1,branch2,branch3,branch4],dim=1)

        return x

class reduct_a(nn.Module):
    def __init__(self,
            in_channels=288,
            out_channels=768
            ):
        super(reduct_a,self).__init__()
        channels, kernel_sizes, strides, paddings = get_reducta()
        self.branch1 = ConvSeq(
                in_channels=in_channels,
                out_channellist=channels[0],
                kernel_sizelist=kernel_sizes[0],
                stridelist=strides[0],
                paddinglist=paddings[0],
                convname='reduct_a_branch1'
                )
        self.branch2 = ConvSeq(
                in_channels=in_channels,
                out_channellist=channels[1],
                kernel_sizelist=kernel_sizes[1],
                stridelist=strides[1],
                paddinglist=paddings[1],
                convname='reduct_a_branch2'
                )
        self.branch3 = nn.Sequential(
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=0)
                )
    
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        x = torch.cat([branch1,branch2,branch3],dim=1)

        return x

def get_inceptionv3_b():
    kernel_sizes = [[1,(1,7),(7,1)],[1,(7,1),(1,7),(7,1),(1,7)]]
    strides = [[1,1,1],[1,1,1,1]]
    paddings = [[0,(0,3),(3,0)],[0,(3,0),(0,3),(3,0),(0,3)]]

    return kernel_sizes,strides,paddings
    

class inceptionv3_b(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            mid_channels
            ):
        super(inceptionv3_b,self).__init__()
        
        kernel_sizes, strides, paddings = get_inceptionv3_b()
        pool_channels = out_channels - 192*3 
        self.branch1 = ConvBlock(
                in_channels=in_channels,
                out_channels=192,
                kernel_size=1,
                stride=1
                )
        self.branch2 = ConvSeq(
                in_channels=in_channels,
                out_channellist=[mid_channels,mid_channels,192],
                kernel_sizelist=kernel_sizes[0],
                stridelist=strides[0],
                paddinglist=paddings[0],
                convname='inceptionv3_b_branch2'
                )
        self.branch3 = ConvSeq(
                in_channels=in_channels,
                out_channellist=[mid_channels,mid_channels,mid_channels,192],
                kernel_sizelist=kernel_sizes[1],
                stridelist=strides[1],
                paddinglist=paddings[1],
                convname='inceptionv3_b_branch3'
                )
        self.branch4 = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=3,
                    stride=1,
                    padding=1
                    ),
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=pool_channels,
                    kernel_size=1,
                    stride=1
                    )
                )

    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        x = torch.cat([branch1,branch2,branch3,branch4],dim=1)

        return x 

def get_reductb():
    channels = [[192,320],[192,192,192,192]]
    kernel_sizes = [[1,3],[1,(7,1),(1,7),3]]
    strides = [[1,2],[1,1,1,2]]
    paddings = [[0,0],[0,(3,0),(0,3),0]]

    return channels,kernel_sizes,strides,paddings

        
class reduct_b(nn.Module):
    def __init__(self,in_channels=768):
        super(reduct_b,self).__init__()

        channels, kernel_sizes, strides, paddings = get_reductb()

        self.branch1 = ConvSeq(
                in_channels=in_channels,
                out_channellist=channels[0],
                kernel_sizelist=kernel_sizes[0],
                stridelist=strides[0],
                paddinglist=paddings[0],
                convname='reductb_branch1'
                )
        self.branch2 = ConvSeq(
                in_channels=in_channels,
                out_channellist=channels[1],
                kernel_sizelist=kernel_sizes[1],
                stridelist=strides[1],
                paddinglist=paddings[1],
                convname='reductb_branch2'
                )
        self.branch3 = nn.Sequential(
                nn.MaxPool2d(
                    kernel_size=3,
                    stride=2,
                    padding=0)
                )
    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        x = torch.cat([branch1,branch2,branch3],dim=1)

        return x

def get_inceptionv3_c():
    channels = [[320],[384],[448,384],[192]]
    kernel_sizes = [[1],[1],[1,3],[3]]
    strides = [[1],[1],[1,1],[1]]
    paddings = [[0],[0],[0,1],[1]]

    return channels,kernel_sizes,strides,paddings


class Conv1331cat(nn.Module):
    def __init__(self,in_channels):
        super(Conv1331cat,self).__init__()
        
        self.branch1 = ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(1,3),
                stride=1,
                padding=(0,1)
                )
        self.branch2 = ConvBlock(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3,1),
                stride=1,
                padding=(1,0)
                )

    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        x = torch.cat([branch1,branch2],dim=1)

        return x
        


class inceptionv3_c(nn.Module):
    def __init__(self,
            in_channels,
            out_channels
            ):
        super(inceptionv3_c,self).__init__()
        
        channels, kernel_sizes, strides, paddings = get_inceptionv3_c()
        self.branch1 = ConvBlock(
                in_channels=in_channels,
                out_channels=channels[0][0],
                kernel_size=kernel_sizes[0][0],
                stride=strides[0][0]
                )
        self.branch2 = ConvSeq(
                in_channels=in_channels,
                out_channellist=channels[1],
                kernel_sizelist=kernel_sizes[1],
                stridelist=strides[1],
                paddinglist=paddings[1],
                convname='inceptionv3_c_branch2'
                )
        self.branch2_1331 = Conv1331cat(in_channels=channels[1][0])
        self.branch3 = ConvSeq(
                in_channels=in_channels,
                out_channellist=channels[2],
                kernel_sizelist=kernel_sizes[2],
                stridelist=strides[2],
                paddinglist=paddings[2],
                convname='inceptionv3_c_branch3'
                )
        self.branch3_1331 = Conv1331cat(in_channels=channels[2][1])
        self.branch4 = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=kernel_sizes[3][0],
                    stride=strides[3][0],
                    padding=paddings[3][0]
                    ),
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=channels[3][0],
                    kernel_size=1,
                    stride=1
                    )
                )

    def forward(self,x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch2 = self.branch2_1331(branch2)
        branch3 = self.branch3(x)
        branch3 = self.branch3_1331(branch3)
        branch4 = self.branch4(x)
        x = torch.cat([branch1,branch2,branch3,branch4],dim=1)

        return x 




























