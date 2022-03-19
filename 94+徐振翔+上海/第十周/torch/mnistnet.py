import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import time

# 加入gpu
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
# print("是否使用GPU：", use_gpu)
# device = "cpu"


# 整体框架
class Model:
    """
    net 定义的神经网络
    cost 损失函数
    optimist 优化算法
    """

    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)

    # 损失函数
    def create_cost(self, cost):
        support_cost = {
            "CROSS_ENTROPY": nn.CrossEntropyLoss(),  # 交叉熵
            "MSE": nn.MSELoss()  # MSE
        }
        return support_cost[cost]

    # 优化算法
    def create_optimizer(self, optimizer, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),  # 随机梯度下降
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            # Adam是SGDM和RMSProp的结合，它基本解决了之前提到的梯度下降的一系列问题，比如随机小样本、自适应学习率、容易卡在梯度较小点等问题
            'RMSprop': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)  # 自适应梯度下降，自适应梯度下降，加入了迭代衰减，
        }
        return support_optim[optimizer]

    # 训练函数
    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            start_time = time.time()
            # 遍历训练集中的数据
            for i, data in enumerate(train_loader, 0):
                # print(type(i), type(data))
                # 准备训练数据与标签
                inputs, labels = data
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels, requires_grad=False).to(device)
                # 梯度清零
                self.optimizer.zero_grad()
                # 正向传播
                outputs = self.net(inputs)
                # 求loss
                loss = self.cost(outputs, labels).to(device)
                # 反向传播求梯度
                loss.backward()
                # 根据学习率和优化方法更新所有参数
                self.optimizer.step()
                # 当前数据的整体损失
                running_loss += loss.item()
                # if i % 100 == 0:
                #     # 打印一批数据（100张）的总损失
                #     print("[epoch:%d,%.2f%%] loss:%.3f]" %
                #           (epoch + 1, (i + 1) * 100. / len(train_loader), running_loss / 100))
                #     running_loss = 0.0
            duration = time.time() - start_time
            print("[epoch:%d,%.2f%%] loss:%.3f]" % (epoch + 1, (i + 1) * 100. / len(train_loader), running_loss / 100),
                  "Training duration:%.4f" % duration)
            running_loss = 0.0

    # 评估模型,测试函数
    def evaluate(self, test_loader):
        print('-' * 20)
        print('Evaluating ...')
        correct = 0
        total = 0
        # 测试、预测时不需要算梯度
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                # inputs = Variable(inputs, requires_grad=True).to(device)
                # labels = Variable(labels, requires_grad=False).to(device)
                # inputs.to(device)
                # labels.to(device)
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels, requires_grad=False).to(device)
                # print("inputs:", inputs.device)
                # print("labels:", labels.device)
                # 输出网络预测结果矩阵
                outputs = self.net(inputs)
                # 最大值索引获取预测结果的索引值
                # torch.argmax(outputs, dims=1)
                # outputs:用于分类的数据 dims=1:求每一行的最大列表;dim=0:求每一列的最大行标
                predict = torch.argmax(outputs, dim=1)
                # 数据总量叠加
                total += labels.size(0)  # size用法同np.shape
                # 累加正确数量，并转为int存储，此时predict是矩阵
                correct += (predict == labels).sum().item()
        print("测试集合数量:%d,模型的精度:%.2f%%" % (total, correct / total * 100))


# 数据加载及预处理
def minst_load_data():
    # 数据导入，图片转换，
    # Compose：即组合几个变换方法，按顺序变换相应数据。
    # 其中torchscript为脚本模块，用于封装脚本跨平台使用，若需要支持此情况，需要使用torch.nn.Sequential，而不是compose
    # 先应用ToTensor()使[0-255]变换为[0-1]，再应用Normalize自定义标准化
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0, ], [1, ])])
    # 下载训练数据,根目录：./data;train=True:从训练集下载数据,否则从测试集下载数据;download:是否下载;transform:转换方法
    trainset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # 双进程下载数据，并打乱(shuffle=True)后分成32batch
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    # 下载测试数据
    testset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader


"""
定义神经网络结构
该神经网络需要集成torch.nn.Model,并在初始化函数中创建网络需要包含的层，并实现forward函数完成前向计算
网络的反向计算由自动求导机制处理
通常 需要训练的层 写在init函数中，将 参数不需要训练的层 在forward方法里调用对应的函数来实现相应的层
"""


# Mnist 模型
class MnistNet(torch.nn.Module):
    def __init__(self):
        # 使用父类的初始化进行初始化
        super(MnistNet, self).__init__()
        # 设计网络结构
        # 输入层
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        # 隐层
        self.fc2 = torch.nn.Linear(512, 200)
        self.fc3 = torch.nn.Linear(200, 10)

    # 前向传播
    def forward(self, x):
        # 数据预处理,同np的resize
        x = x.view(-1, 28 * 28)
        # 激活函数
        # 输入层，经过relu的激活函数
        x = F.relu(self.fc1(x))
        # 隐层，经过relu的激活函数
        x = F.relu(self.fc2(x))
        # 输出层，经过softmax
        # dim = 1：对每一行的所有元素进行softmax运算，并使得每一行所有元素和为1。
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    t_s = time.time()
    # 数据导入及预处理
    train_loader, test_loader = minst_load_data()
    print("train size:{},test size:{}".format(len(train_loader), len(test_loader)))
    print(type(train_loader), type(test_loader))
    # 初始化网络
    net = MnistNet().to(device)
    # 实例化模型
    model = Model(net, 'CROSS_ENTROPY', "RMSprop")
    # 训练
    model.train(train_loader, 10)
    # model.train(train_loader, 3)
    # 评估训练结果
    model.evaluate(test_loader)
    t_e = time.time()
    print("总用时:%.4f" % (t_e - t_s))
