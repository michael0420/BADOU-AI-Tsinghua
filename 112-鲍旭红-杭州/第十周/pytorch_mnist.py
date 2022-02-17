#!/usr/bin/env python
# encoding: utf-8
'''
@author: 醉雨成风
@contact: 573878341@qq.com
@software: python
@file: minist.py
@time: 2022/1/28 17:26
@desc:
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import cv2

#下载训练集
train_dataset = datasets.MNIST(root='./mnist_datasets',
                               train=True,transform=transforms.ToTensor(),download=True)

#下载测试集
test_dataset = datasets.MNIST(root='./mnist_datasets',
                               train=False,transform=transforms.ToTensor(),download=True)

batch_size = 32

#装载训练集
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

#装载测试集
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

#数据预览
images,labels = next(iter(train_dataset))
img = torchvision.utils.make_grid(images)
img = img.numpy().transpose(1,2,0)
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img * std + mean
print(labels)
cv2.imshow("img",img)
cv2.waitKey(0)


#搭建神经网络
# 卷积层用torch.nn.Conv2d
# 激活层用torch.nn.Relu
# 池化层用torch.nn.MaxPool2d
# 全链接层用torch.nn.Linear


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1,6,3,1,2),nn.ReLU(),nn.MaxPool2d(2,2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(16*5*5,120),nn.BatchNorm1d(120),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120,84),nn.BatchNorm1d(84),nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(84,10))


    def forward(self,x):
        '''
        前向传播
        :param x:
        :return:
        '''
        x = self.conv1(x)
        x = self.conv2(x)
        # 对参数实现扁平化（便于后面全连接层输入），相当于reshape
        x = x.view(x.size()[0],-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def train(net):
    Lr = 0.001

    # 损失函数使用交叉熵
    crtertion = nn.CrossEntropyLoss()
    # 优化函数使用 Adam 自适应优化算法
    optimizer = optim.Adam(net.parameters(), lr=Lr)
    epotchs = 10
    for epotch in range(epotchs):
        sum_loss = 0.0
        for i,data in enumerate(train_loader):
            inputs,labels = data
            inputs,labels = Variable(inputs).cuda(),Variable(labels).cuda()
            optimizer.zero_grad()#将梯度归零
            outputs = net(inputs)#将数据传入网络进行前向传播
            loss = crtertion(outputs,labels)#得到损失函数
            loss.backward()#反向传播
            optimizer.step()#通过梯度做一步参数更新
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[epotch: %d,i : %d] loss :%.03f'%(epotch+1,i+1,sum_loss/100))
                sum_loss = 0.0




def net_test(net):
    net.eval()#转换为测试模型
    correct = 0
    total = 0
    for data_test in test_loader:
        images, labels = data_test
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        output_test = net(images)
        _,predicted = torch.max(output_test,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
        print('correct:',correct)
        print('test acc:{0}'.format(correct.item() / len(test_dataset)))



if __name__ == '__main__':
    net = LeNet().to(device)
    train(net)#训练模型
    net_test(net)






