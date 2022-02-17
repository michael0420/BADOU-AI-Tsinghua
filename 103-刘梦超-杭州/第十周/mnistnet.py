#!/usr/bin/env python 
# coding:utf-8

import torch
import torchvision.datasets
from torch import nn, optim
from torchvision import transforms
import torch.nn.functional as F


# 加载并处理数据
def mnist_load_data():
    # 图片变换操作,tensor数据归一化到[0.0,1.0],再对其通道自定义处理,为了加快收敛速度
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0], [1])])
    # 下载训练数据
    trainset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    # 两个进程处理数据,打乱顺序分32批
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        # 父类初始化
        super(MnistNet, self).__init__()
        # 网络结构
        self.fc1 = torch.nn.Linear(784, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        # 改变输入数据的形状
        x = x.view(-1, 784)
        # 过激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 按行计算
        x = F.softmax(self.fc3(x), dim=1)
        return x


class Model:
    def __init__(self, net, cost, optimizer):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimizer)

    # 损失函数
    def create_cost(self, cost):
        support_cost = {
            "CROSS_ENTROPY": nn.CrossEntropyLoss(),
            "MSE": nn.MSELoss()
        }
        return support_cost[cost]

    # 优化算法
    def create_optimizer(self, optimizer):
        support_optimizer = {
            "SGD": optim.SGD(self.net.parameters(), lr=0.1),
            "ADAM": optim.Adam(self.net.parameters(), lr=0.01),
            "RMSP": optim.RMSprop(self.net.parameters(), lr=0.001)
        }
        return support_optimizer[optimizer]

    # 训练
    def train(self, train_loader, epoches):
        # 训练次数
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # 训练数据和标签
                inputs, labels = data
                # 梯度清零
                self.optimizer.zero_grad()
                # 正向传播
                outputs = self.net(inputs)
                # 损失函数
                loss = self.cost(outputs, labels)
                # 反向传播
                loss.backward()
                # 更新网络参数
                self.optimizer.step()
                # 部分数据的损失总量
                running_loss += loss.item()
                if i % 100 == 0:
                    print("epoch:%d,当前的进度:%.2f%%,loss:%.3f" % (
                        (epoch + 1), (i + 1) / len(train_loader) * 100, running_loss / 100))
                    running_loss = 0.0

    # 评估
    def evaluate(self, test_loader):
        correct = 0
        total = 0
        # 以下tensor不需要求梯度
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                # 输出结果
                outputs = self.net(inputs)
                # 每行最大值索引
                predict = torch.argmax(outputs, 1)
                # 总数据量
                total += labels.size(0)
                # 如果相同则为1,不同为0,累加,再将tensor转为int
                correct += (predict == labels).sum().item()
        print("该模型的精度:%.2f%%" % (correct / total * 100))


if __name__ == '__main__':
    # 加载数据
    train_loader, test_loader = mnist_load_data()
    # 实例化模型
    net = MnistNet()
    model = Model(net, "CROSS_ENTROPY", "RMSP")
    # 训练
    model.train(train_loader, 3)
    # 评估
    model.evaluate(test_loader)
