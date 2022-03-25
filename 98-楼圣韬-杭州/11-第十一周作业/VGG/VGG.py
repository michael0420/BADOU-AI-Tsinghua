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
import torch.utils.data
import numpy as np
import cv2
import matplotlib.pyplot as plt
from misc import progress_bar
import torch.backends.cudnn as cudnn

NUM_CLASSES=10
# cifar-10 的十个分类类别
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG11():
    return VGG('VGG11')


def VGG13():
    return VGG('VGG13')


def VGG16():
    return VGG('VGG16')


def VGG19():
    return VGG('VGG19')



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
                progress_bar(batch_num, len(self.test_loader), 'Loss: %.4f | ACC:%.3f%%(%d/%d)'
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
k=Model(arg,VGG16())
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