"""
    利用pytorch框架实现mnist手写数字识别
"""
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class Model:
    def __init__(self, net, cost, optimizer):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimizer, lr=0.001)

    def create_cost(self, cost):
        """构建代价函数，支持交叉熵和均方误差两种损失函数
        :param cost: str,选择对应的损失函数
        :return:
        """
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss(),
        }

        return support_cost[cost]

    def create_optimizer(self, optimizer, **rests):
        """构建网络模型的优化算法
        :param optimizer: str,选择对应的优化器
        :param rests: dict,提供优化器的各项关键字参数
        :return: 返回优化器
        """
        support_optimizer = {
            'SGD': optim.SGD(self.net.parameters(), **rests),
            'ADAM': optim.Adam(self.net.parameters(), **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), **rests),
        }
        return support_optimizer[optimizer]

    def train(self, train_loader, epochs=3):
        """网络模型的训练接口
        :param train_loader:数据迭代器
        :param epochs:训练轮次
        :return:
        """
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.net = self.net.to(device)
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()  # 梯度清零，避免梯度信息累加
                predicted = self.net(inputs)  # 网络前向传播
                loss = self.cost(predicted, labels)  # 计算损失函数值
                loss.backward()  # 梯度反向传播
                self.optimizer.step()  # 更新网络参数
                running_loss += loss.item()

                # 训练信息显示
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0
        print("网络训练完成！")

    def evaluate(self, test_loader):
        """评估网络的性能
        :param test_loader:
        :return:
        """
        print("Evaluating...")
        correct = 0
        total = 0
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.net = self.net.to(device)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, dim=1)
                total += labels.shape[0]
                correct += (labels == predicted).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100.0 * correct / total))

    def visualize_samples(self, samples=6):
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.net = self.net.to(device)
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, dim=1)

                inputs = inputs.detach().cpu().numpy()
                inputs *= 255
                inputs = inputs.astype("uint8")
                inputs = inputs.reshape(32, 28, 28)
                labels = labels.detach().cpu().numpy()
                predicted = predicted.detach().cpu().numpy()

                # 展示部分样本结果
                plt.figure()
                for i in range(samples):
                    plt.subplot(2, int(samples / 2), i + 1)
                    plt.imshow(inputs[i], cmap='gray')
                    plt.title("[%d \ %d]" % (labels[i], predicted[i]))
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                plt.savefig("./predicted_samples.png")
                plt.show()
                break


def mnist_load_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0, ], [1, ])
        ]
    )

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          transform=transform, download=True)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


if __name__ == "__main__":
    # mnist_load_data()
    # # train_loader, test_loader = mnist_load_data()
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader,5)
    model.evaluate(test_loader)
    model.visualize_samples()
