import torch
from torch import nn

def get_resnet():
    net_layers = [3,4,6,3]
    net_channels = [64,128,256,512]

    return net_layers,net_channels



class conv(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=1,
            groups=1,
            bias=False,
            use_bn=True,
            bn_eps=1e-5,
            use_act=True,
            ):
        super(conv,self).__init__()
        self.use_bn = use_bn
        self.use_act = use_act
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
        if self.use_act:
            self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.use_act:
            x = self.act(x)
        return x

class IdentityBlock(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            ):
        super(IdentityBlock,self).__init__()
        self.channels_reset = in_channels!=out_channels
        if self.channels_reset:
            self.set_conv = conv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    use_act=False
                    )
        mid_channels = in_channels
        self.conv1 = conv(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                )
        self.conv2 = conv(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1
                )
        self.conv3 = conv(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                use_bn=False,
                use_act=False
                )
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        if self.channels_reset:
            input_x = self.set_conv(x)
        else:
            input_x = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = input_x + x
        x = self.act(x)
        return x
        
class ConvBlock(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            ):
        super(ConvBlock,self).__init__()

        mid_channels = in_channels
        self.convl1 = conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=2,
                padding=0,
                use_bn=True,
                use_act=False
                )
        self.convr1 = conv(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=2,
                padding=0,
                )
        self.convr2 = conv(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                stride=1,
                padding=1
                )
        self.convr3 = conv(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                use_bn=False,
                use_act=False
                )
        self.act = nn.ReLU(inplace=True)

    def forward(self,x):
        lx = self.convl1(x)
        rx = self.convr1(x)
        rx = self.convr2(rx)
        rx = self.convr3(rx)
        x = lx + rx
        x = self.act(x)
        return x

class resnet50_inlayers(nn.Module):
    def __init__(self,in_channels,out_channels=64):
        super(resnet50_inlayers,self).__init__()
        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                dilation=1,
                groups=1,
                bias=False
                )
        self.bn = nn.BatchNorm2d(num_features=out_channels,eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.max = nn.MaxPool2d(
                kernel_size=3,
                stride=2,padding=0,ceil_mode=True)
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.max(x)
        return x

class resnet50(nn.Module):
    def __init__(self,class_num=1000,in_size=(224,224),in_channels=3):
        super(resnet50,self).__init__()
        self.class_num = class_num
        self.in_size = in_size
        net_layers, net_channels = get_resnet()
        self.channelslists = [[ch]*net_layers[ch_i] for ch_i,ch in enumerate(net_channels)]
        self.inlayers = resnet50_inlayers(
                in_channels=in_channels)
        in_channels = 64
        self.net = nn.Sequential()
        for i,block in enumerate(self.channelslists):
            net_ = nn.Sequential()
            for j,channels in enumerate(block):
                identity = False if j==0 and i!=0 else True
                out_channels = channels*4
                if identity:
                    net_.add_module(f'Big{i}smallblock{j}',IdentityBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ))
                else:
                    net_.add_module(f'Big{i}smallblock{j}',ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        ))
                in_channels = out_channels 
            self.net.add_module(f'BigBlock{i}',net_)
        self.avg = nn.AvgPool2d(
                kernel_size=7,
                stride=1)
        self.fc = nn.Linear(in_channels,self.class_num)
        self.softmax = nn.Softmax(dim=1)

        
    def forward(self,x):
        x = self.inlayers(x)
        x = self.net(x)
        x = self.avg(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        x = self.softmax(x)

        return x

def main_test(device,batch_size,class_num):

    x = torch.randn((batch_size,3,224,224),device=device)
    model = resnet50(class_num).to(device)
    model.eval()
    pred = model(x)
    pred_class = argmax(pred)
    print('pred_class:',pred_class)
    print('success test')
    
if __name__=='__main__':
    """
    pytorch 实现resnet50
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 8
    class_num = 10
    main_test(device,batch_size,class_num)
