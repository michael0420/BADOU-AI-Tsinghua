import torch
from torch import nn
from common import ConvBlock
from inceptions import inceptionv3_a, reduct_a, inceptionv3_b, reduct_b, inceptionv3_c
def get_v3_headconv():
    """
    inceptionv3 模块结构数据
    """
    strides = [[2,1,1],[2],[1,1],[2]]
    paddings = [[0,0,1],[0],[0,0],[0]]
    channels = [[32,32,64],[64],[80,192],[192]]
    kernel_sizes = [[3,3,3],[3],[1,3],[3]]

    return channels,strides,paddings,kernel_sizes

def get_v3_body():
    """
    返回 inceptionv3 模块部分结构参数
    """
    channels = [[256,288,288],
                [768,768,768,768,768],
                [1280,2048,2048]]
    b_mid_channels = [128,160,160,192]

    return channels,b_mid_channels

class inceptionv3(nn.Module):
    def __init__(self,class_num,in_size=(299,299),in_channels=3):
        super(inceptionv3,self).__init__()
        self.class_num = class_num
        channels, strides, paddings, kernels = get_v3_headconv()
        body_channels, b_mid_channels = get_v3_body()

        self.headconvnet = nn.Sequential()
        for i in range(4):
            for j,(out_channels,stride,padding,kernel_size) in enumerate(zip(channels[i],strides[i],paddings[i],kernels[i])):
                if i!=1 and i!=3:
                    self.headconvnet.add_module(f'conv{i}_{j}',ConvBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        ))
                else:
                    self.headconvnet.add_module(f'maxpool{i}',nn.MaxPool2d(
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        ceil_mode=True
                        ))
                in_channels = out_channels
        self.inception_a = nn.Sequential()
        for i,out_channels in enumerate(body_channels[0]):
            self.inception_a.add_module(f'inceptiona{i}',inceptionv3_a(
                        in_channels=in_channels,
                        out_channels=out_channels
                        ))
            in_channels = out_channels
        self.reduct_a = reduct_a()
        in_channels = 768
        self.inception_b = nn.Sequential()
        for i,out_channels in enumerate(body_channels[1][1:]):
            self.inception_b.add_module(f'inceptionb{i}',inceptionv3_b(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        mid_channels=b_mid_channels[i]
                        ))
            in_channels = out_channels
        self.reduct_b = reduct_b()
        self.inception_c = nn.Sequential()
        in_channels = 1280
        for i,out_channels in enumerate(body_channels[2][1:]):
            self.inception_c.add_module(f'inceptionc{i}',inceptionv3_c(
                        in_channels=in_channels,
                        out_channels=out_channels
                        ))
            in_channels = out_channels
        self.avgpool = nn.AvgPool2d(
                kernel_size=8,
                stride=1,
                padding=0
                )
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(2048,self.class_num)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self,x):
        x = self.headconvnet(x)
        x = self.inception_a(x)
        x = self.reduct_a(x)
        x = self.inception_b(x)
        x = self.reduct_b(x)
        x = self.inception_c(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        

        return x
        
        

        


def test_main(device,batch_size,class_num):

    x = torch.randn((batch_size,3,299,299),device=device)

    model = inceptionv3(class_num).to(device)
    model.eval()
    pred = model(x)
    print(pred.shape)



if __name__=='__main__':
    """
    torch 实现inceptionv3分类神经网络
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    class_num = 10
    batch_size = 8
    test_main(device,batch_size,class_num)


