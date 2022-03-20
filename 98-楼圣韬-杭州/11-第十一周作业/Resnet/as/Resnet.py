from ast import Mod, parse
from typing_extensions import Self
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import torch.backends.cudnn as cudnn
import torch.utils.data
import numpy as np
import cv2
import matplotlib.pyplot as plt
from misc import progress_bar
import math

NUM_CLASSES=10
# cifar-10 的十个分类类别
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

k=0;

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3 convolution with padding
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        x += residual
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10):   # bottleneck [3,4,6,3]
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # block.expansion=4

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):  # bacth make (model,dstplanes,)
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)


class Model():
    def __init__(self,config,Model) -> None:
        self.cuda = config.cuda
        if self.cuda:
            self.device=torch.device('cuda')
            cudnn.benchmark = True
        else:
            self.device = torch('cpu')
        self.lr = config.lr
        self.epochs = config.epoch
        self.train_batch_size = config.trainBatchSize
        self.test_batch_size = config.testBatchSize
        self.train_loader = None
        self.test_loader = None
        self.model=Model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(),lr=self.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[75,150],gamma=0.5)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def CIFAR10_load_data(self):
        # 原论文采用输入图片为224*224 ，此处为减小计算量，使用原数据图大小32*32
        train_transform=transforms.Compose(
            [# transforms.Resize([224,224]), 
            transforms.RandomHorizontalFlip(), # 以一定的概率水平翻转图像，默认为0.5
            transforms.ToTensor()]   # 将数据转化为张量
        )
        test_transform=transforms.Compose(
            [# transforms.Resize([224,224]),
            transforms.ToTensor()]
        )
        train_set=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=train_transform)
        self.train_loader=torch.utils.data.DataLoader(dataset=train_set,batch_size=self.train_batch_size,shuffle=True)
        test_set=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=test_transform)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_set,batch_size=self.test_batch_size,shuffle=True)
        return train_set,test_set,self.train_loader,self.test_loader


    def train(self):
        print("train:")
        self.model.train() # 进入训练模式
        train_loss=0
        train_correct=0
        total = 0

        for batch_num,(data,target) in enumerate(self.train_loader):
            data,target = data.to(self.device),target.to(self.device)
            self.optimizer.zero_grad() # 梯度清零
            output = self.model(data)
            loss=self.criterion(output,target)
            loss.backward()
            self.optimizer.step()
            train_loss +=loss.item() # item()将单个张量转化为数值
            prediction = torch.max(output,1) # 返回两个值，第一个是最大值，第二个是最大值的索引
            total +=target.size(0)
            train_correct+=np.sum(prediction[1].cpu().numpy()==target.cpu().numpy())
            progress_bar(batch_num,len(self.train_loader),'LOSS:%.4f|ACC: %.3f%%(%d/%d)'
                        %(train_loss/(batch_num+1),100.*train_correct/total,train_correct,total))
        return train_loss,train_correct /total

    def test(self):
        print("test:")
        self.model.eval()  # 进入测试模式，部分操作不执行，例如Dropout
        test_loss =0
        test_correct =0
        total =0
        with torch.no_grad(): # 不执行记忆
            for batch_num,(data,target) in enumerate(self.test_loader):
                data,target = data.to(self.device),target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output,target)
                test_loss +=loss.item() # item()将单个张量转化为数值
                prediction = torch.max(output,1)
                total+=target.size(0)
                test_correct+=np.sum(prediction[1].cpu().numpy()==target.cpu().numpy())
                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_num + 1), 100. * test_correct / total, test_correct, total))
        return test_loss,test_correct /total

    def save(self):  # 输出模型权重
        model_out_path= "model.pth"
        torch.save(self.model,model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

def parserArg():
    parser = argparse.ArgumentParser(description='cifar-10 with PyTorch')
    parser.add_argument('--lr',default=0.01,type=float,help='learning rate')
    parser.add_argument('--epoch',default=200,type=int,help='number of epoches tp train for')
    parser.add_argument('--trainBatchSize',default=100,type=int,help='training batch size')
    parser.add_argument('--testBatchSize',default=100,type=int,help='testing batch size')
    parser.add_argument('--cuda',default=torch.cuda.is_available(),type=bool,help='whether cuda is availiable')
    args = parser.parse_args()
    return args


def printTensor(images):   # 可视化的关键在于原数据是[channel,width,height]，而图像是[width,height,channel]
    t=np.array(images*255,dtype=np.uint8)
    _,width,height=t.shape
    img=t.reshape(-1,width*height)
    r=img[0,:].reshape(width,height)
    g=img[1,:].reshape(width,height)
    b=img[2,:].reshape(width,height)
    dstimg=np.zeros((32,32,3),dtype=np.uint8)
    dstimg[:,:,0]=r
    dstimg[:,:,1]=g
    dstimg[:,:,2]=b
    cv2.imshow('img',dstimg)
    cv2.imwrite('.\i.jpg',dstimg)
    cv2.waitKey(0)



def testData1(data):
    data1=list(enumerate(data,0))
    print(data1[0])
    for i,data2 in enumerate(data,0):
        print(i,"\n",data2)
        images,labels=data2
        print("__________________________________________")
        print(labels)
        break
    return images

def testbar():  # 测试progress_bar
    for i in range(300000):
        progress_bar(i,300000)

arg=parserArg()
k=Model(arg,resnet50())
train_set,test_set,train_loader,test_loader=k.CIFAR10_load_data()
accuracy=0
for epoch in range(1,k.epochs+1):
    k.scheduler.step(epoch)
    print("\n===> epoch: %d/200"%epoch)
    train_result = k.train()
    print(train_result)
    test_result = k.test()
    accuracy = max(accuracy,test_result[1])
    if epoch == k.epochs:
        print("===> BEST ACC. PERFORMANCE: %.3f%%"%(accuracy*100))
        k.save()
#printTensor(testData1(train_set))
#for batch_num, (data, target) in enumerate(train_loader):
#    print(batch_num)
#    print((data,target))
#    print(data.size())
#    print(target.size())
#    break
#testbar()