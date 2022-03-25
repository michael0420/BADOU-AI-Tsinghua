import torch
from torch import nn
from torch.nn import functional as F

def get_alexnet():
    """
    alexnet 模型结构数据
    """
    channels = [[96],[256],[384,384,256]]
    kernel_sizes = [[11],[5],[3,3,3]]
    strides = [[4],[1],[1,1,1]]
    paddings = [[0],[2],[1,1,1]]
    return channels,kernel_sizes,strides,paddings

class AlexConvBlock(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            use_lrn
            ):
        super(AlexConvBlock,self).__init__()
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=1,
                groups=1,
                bias=True
                )
        self.act = nn.ReLU()
        self.use_lrn = use_lrn

    def forward(self,x):
        x = self.conv(x)
        x = self.act(x)
        if self.use_lrn:
            x = F.local_response_norm(x,size=5,k=2.0)
        return x


class alexnet(nn.Module):
    def __init__(self,class_num):
        super(alexnet,self).__init__()
        
        channels, kernel_sizes, strides, paddings = get_alexnet()
        self.fc_input = channels[-1][-1]*6*6
        self.fc_mid = 4096
        self.fc_out = class_num
        self.net = nn.Sequential()
        for i in range(len(channels)):
            net_ = nn.Sequential()
            use_lrn = True if i in [0,1] else False
            for j in range(len(channels[i])):
                if i==0 and j==0:
                    in_channels = 3
                net_.add_module(f'conv_{i}{j}',AlexConvBlock(
                    in_channels,
                    channels[i][j],
                    kernel_sizes[i][j],
                    strides[i][j],
                    paddings[i][j],
                    use_lrn
                    ))
                in_channels = channels[i][j]
            self.net.add_module(f'max{i}',nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=0,
                ceil_mode=True))
            self.net.add_module(f'net{i}',net_)
            self.fcs = nn.Sequential(
                    nn.Linear(self.fc_input,self.fc_mid),
                    nn.Linear(self.fc_mid,self.fc_mid),
                    nn.Linear(self.fc_mid,self.fc_out)
                    )
    def forward(self,x):
        x = self.net(x)
        x = x.view(x.size(0),-1)
        x = self.fcs(x)
        return x





def main(batch_size,device,class_num):
    """
    网络结构
    """

    x = torch.randn((batch_size,3,224,224)).to(device)
    model = alexnet(class_num).to(device)
    y = model(x)
    print(y.shape)




if __name__=='__main__':
    """
    torch 实现alexnet网络训练和推理 
    """
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_num = 1000
    main(batch_size,device,class_num)
