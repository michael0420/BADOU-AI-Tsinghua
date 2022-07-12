"""
练习使用pytorch定义一个神经网络，并且试着使用MNIST数据集训练和测试它
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils import data


class MyModel:
    """
    定义模型
    """

    def __init__(self, mynet, mycost, myoptimist):
        """
        初始化模型属性
        :param mynet: 传入定义好的神经网络
        :param mycost: 代价函数名称，type:str
        :param myoptimist: 优化器名称，type:str
        """

        self.net = mynet
        self.cost = self.create_cost(mycost)
        self.optimizer = self.create_optimizer(myoptimist)

    def create_cost(self, cost):
        """
        根据输入返回对应的代价函数
        :param cost: 代价函数名称，type:str
        :return: 根据名称索引的代价函数
        """

        cost_bank = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return cost_bank[cost]

    def create_optimizer(self, optimist, **rests):
        """
        根据输入返回对应的优化器
        :param optimist: 优化器名称，type:str
        :param rests: 需特别定义的优化器的其他参数，如学习率lr(如果不指定lr则使用本方法内置传入优化器的lr)
        :return: 根据名称索引优化器
        """

        optim_bank = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return optim_bank[optimist]

    def train(self, loaded_train_data, epoches=5):
        """
        定义训练过程
        :param loaded_train_data: 载入供训练的数据集
        :param epoches: 每个数据被遍历次数，default=5
        """

        for epoch in range(epoches):
            running_loss = 0.0  # 每次遍历前重新初始化loss值
            for i, _data in enumerate(loaded_train_data, start=0):
                inputs, labels = _data  # 切分数据集为：数据(inputs),标签(labels)

                self.optimizer.zero_grad()  # 每个batch都要重新初始化梯度为0，消除前一个batch保留梯度的影响

                # 前向传递 + 反向传递 + 权重优化
                outputs = self.net(inputs)  # 前向传递得到在第t轮各权重w(i,t)条件下神经网络的输出
                loss = self.cost(outputs, labels)  # 计算与label对比，计算代价函数值
                loss.backward()  # 计算梯度值
                self.optimizer.step()  # 权重更新

                running_loss += loss.item()  # Tensor.item()方法是将tensor的值转化成python number
                if i % 100 == 0:  # 每100个batch输出一次，同时将running loss清零
                    print(f"[epoch{epoch + 1}, {(i + 1) * 1. / len(loaded_train_data) * 100:2f}%] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, loaded_test_data):
        """
        测试训练效果
        :param loaded_test_data: 载入供测试的数据集
        """

        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # 在测试和预测模式下，不再计算梯度反向传播等
            for _data in loaded_test_data:
                images, labels = _data  # 切分数据集

                # 前向传递 + 分类 + 正确率计算
                outputs = self.net(images)  # 在训练好的模型中传入数据，得到输出
                predicted = torch.argmax(outputs, 1)  # 返回预测最大概率的索引
                total += labels.size(0)  # 计算预测总数
                correct += (predicted == labels).sum().item()  # 计算正确次数，并由Tensor转化成python number

        print(f"Accuracy of the network on the test images: {100 * correct / total}%")


def mnist_load_data():
    """
    下载MNIST数据集并转换处理
    :return: 训练集(train_loader)和测试集(test_loader)
    """

    # 首先规定数据预处理方法：使用torchvision.transforms.Compose()函数将需要的操作顺序串联
    # 先使用torchvision.transforms.ToTensor()将PIL Image转换为tensor，并归一化到[0,1]之间
    # 再使用torchvision.transforms.Normalize()将tensor的数值标准化mean=0,std=1
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])

    # 规定好转换动作(存入transform变量中)后，加载训练集和测试集
    # 下载MNIST数据集到./data目录，train=True代表生成训练集False代表生成测试集，transform参数传入上述规定的转化操作
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # 将训练数据集分成1875个batch(60000个数据，batch_size=32)，shuffle=True打乱数据顺序，num_workers=2使用2个子进程导入数据
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader


class MnistNet(torch.nn.Module):
    """
    定义手写数据集识别神经网络
    """

    def __init__(self):
        """
        定义神经网络所需要的属性，并初始化
        """
        super().__init__()  # 从父类torch.nn.Module继承其属性
        self.fc1 = torch.nn.Linear(28 * 28, 512)  # 定义并初始化第一个全链接层，512个神经节点接受28*28=784个输入
        self.fc2 = torch.nn.Linear(512, 512)  # 第二层全链接层，512个节点
        self.fc3 = torch.nn.Linear(512, 10)  # 最后一个全链接层，即输出层，有10个节点用于多分类对应0~9十个数字

    def forward(self, x):
        """
        定义前馈函数
        :param x: 输入数据
        :return: 前馈输出x
        """

        x = x.view(-1, 28 * 28)  # 相当于numpy的reshape。此处是将输入数据变换成不固定行数，限定为784列，因此第一个参数是-1
        x = F.relu(self.fc1(x))  # 将上一步的x传入fc1层，前经RuLu激活后输出
        x = F.relu(self.fc2(x))  # 在fc2层重复上一步骤的过程
        x = F.softmax(self.fc3(x), dim=1)  # 数据传递至fc3层，输出前经softmax函数处理，沿着行(dim=1)求得softmax
        return x


if __name__ == '__main__':

    net = MnistNet()  # 实例化神经网络
    model = MyModel(net, 'CROSS_ENTROPY', 'RMSP')  # 将神经网络、代价函数、优化器传入模型
    train_loader, test_loader = mnist_load_data()  # 下载、处理数据集
    model.train(train_loader, epoches=5)  # 训练模型
    model.evaluate(test_loader)  # 测试模型
