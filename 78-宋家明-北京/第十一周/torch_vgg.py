import torch
from torch import nn

def get_vgg16net():
    """
    放置vgg 卷积神经网络结构数据
    """
    net_nums = [2,2,3,3,3]
    net_channels = [64,128,256,512,512]
    

    return net_nums,net_channels

# vgg卷积基础单元
class VggConvBlock(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True
            ):
        super(VggConvBlock,self).__init__()
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias
                )
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        x = self.act(x)

        return x

# vgg 神经网络
class vgg16net(nn.Module):
    def __init__(self,class_num):
        super(vgg16net,self).__init__()
        net_nums, net_channels = get_vgg16net()
        self.net_list = [net_nums[i]*[net_channels[i]] for i in range(len(net_nums))]
        self.class_num = class_num
        self.net = nn.Sequential()

        in_channels = 3
        for layerid,layer in enumerate(self.net_list):
            net_ = nn.Sequential()
            for channelid,channel in enumerate(layer):
                net_.add_module(f'conv{layerid}{channelid}',VggConvBlock(in_channels,channel))
                in_channels = channel
            net_.add_module(f'maxpool{layerid}',nn.MaxPool2d(kernel_size=2))
            self.net.add_module(f'layer{layerid}',net_)
        self.fc = nn.Sequential(
                nn.Linear(in_channels*7*7,4096),
                nn.Linear(4096,4096),
                nn.Linear(4096,self.class_num)
                )
        self.softmax = nn.Softmax()
    def forward(self,x):
        x = self.net(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = self.softmax(x)
        return x





def test_main(batch_size,device,class_num):
    """
    测试vgg模型输出预测值
    """
    x = torch.randn((batch_size,3,224,224),device=device)
    model = vgg16net(class_num).to(device)
    pred = model(x)

    print(pred.shape)
    print(pred.argmax(0))


if __name__=='__main__':
    """
    torch 实现vgg 卷积神经网络
    """
    
    batch_size = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_num = 10
    test_main(batch_size,device,class_num)
