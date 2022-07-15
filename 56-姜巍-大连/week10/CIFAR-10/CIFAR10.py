# 使用torchvision和transforms下载并标准化cifar-10数据集
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# 中间过程显示一些数据集中图像
import matplotlib.pyplot as plt
import numpy as np

# 编写神经网络代码所需库
import torch.nn as nn
import torch.nn.functional as F
# 选择优化器所需库
import torch.optim as optim

# 编写transform操作顺序：将数据转换成tensor,然后归一化为[-1,1]之间
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
# 训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
# 测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


def imshow(img):
    """
    显示一些图像
    :param img: 输入张量
    """
    img = img / 2 + 0.5  # 去归一化
    npimg = img.numpy()
    # 上面transform.ToTensor()操作后数据编程CHW[通道靠前模式]，需要转换成HWC[通道靠后模式]才能plt.imshow()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 转置前将排在第0位的Channel(C)放在最后，所以是(1,2,0)
    plt.show()


class Net(nn.Module):
    """定义一个卷积神经网络及前馈函数"""

    def __init__(self):
        """初始化网络：定义卷积层、池化层和全链接层"""

        super().__init__()  # 继承父类属性。P.S. 如果看到super(Net, self).__init__()写法亦可
        self.conv1 = nn.Conv2d(3, 6, 5)  # 使用2套卷积核。输入(B×3×32×32),输出(B×6×28×28)
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化操作，输出时高、宽减半，(B×6×14×14)  (B×16×5×5)
        self.conv2 = nn.Conv2d(6, 16, 5)  # 使用4套卷积核，卷积核大小为5×5。(B×16×10×10)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全链接层。将数据扁平化成一维，共400个输入，120个输出
        self.fc2 = nn.Linear(120, 84)  # 全链接层。120个输入，84个输出
        self.fc3 = nn.Linear(84, 10)  # 全链接层。84个输入，10个输出用于分类

    def forward(self, x):
        """前馈函数，规定数据正向传播的规则"""

        x = self.pool(F.relu(self.conv1(x)))  # 输入 > conv1卷积 > ReLu激活 > maxpool最大池化
        x = self.pool(F.relu(self.conv2(x)))  # > conv2卷积 > ReLu激活 > maxpool最大池化
        # x = torch.flatten(x, 1)  # 如果你不喜欢下一种写法实现扁平化，可以使用这条语句代替
        x = x.view(-1, 16 * 5 * 5)  # 相当于numpy的reshape。此处是将输入数据变换成不固定行数，因此第一个参数是-1，完成扁平化
        x = F.relu(self.fc1(x))  # 扁平化数据 > fc1全链接层 > ReLu激活
        x = F.relu(self.fc2(x))  # > fc2全链接层 > ReLu激活
        x = self.fc3(x)  # > fc3全链接层 > 输出
        return x


if __name__ == '__main__':

    # # 随机输出一个mini batch的训练集图像
    # dataiter_tr = iter(trainloader)  # 取一个batch的训练集数据
    # images_tr, labels_tr = next(dataiter_tr)  # 切分数据和标签
    # imshow(torchvision.utils.make_grid(images_tr))  # 生成网格图
    # print(' '.join(f'{classes[labels_tr[j]]:5s}' for j in range(batch_size)))  # 打印标签值

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检测是否可以在GPU上运行训练过程
    print(f"model will be trained on device: '{device}'")

    net = Net().to(device)  # 实例化神经网络，存入GPU

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # SGDM

    print(f"Now comes the training procedure, batch size is {batch_size}.")

    for epoch in range(5):  # 数据被遍历的次数

        running_loss = 0.0  # 每次遍历前重新初始化loss值
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # 切分数据集

            optimizer.zero_grad()  # 梯度清零，避免上一个batch迭代的影响

            # 前向传递 + 反向传递 + 权重优化
            outputs = net(inputs).to(device)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()

            # 输出日志
            running_loss += loss.item()  # Tensor.item()方法是将tensor的值转化成python number
            if i % 2000 == 1999:  # 每2000个mini batches输出一次
                # print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))  如果python3.6之前版本可以使用这个代码
                print(f'[epoch:{epoch + 1}, mini-batch:{i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training !')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)  # 保存训练好的模型到指定路径

    # # 随机输出一个mini batch的测试集图像
    # dataiter_te = iter(testloader)
    # images_te, labels_te = next(dataiter_te)
    # imshow(torchvision.utils.make_grid(images_te))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels_te[j]] for j in range(batch_size)))

    test_net = Net().to(device)
    test_net.load_state_dict(torch.load(PATH))

    # outputs = net(images_te).to(device)  # 看一下神经网络对上述展示图片的预测结果
    # predicted = torch.max(outputs, 1)[1].to(device)  # torch.max(input, dim)返回按照dim方向的最大值和其索引
    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(batch_size)))

    correct = 0
    total = 0
    # 由于这不是在训练模型，因此对输出不需要计算梯度等反向传播过程
    with torch.no_grad():
        for data in testloader:
            images_pre, labels_pre = data[0].to(device), data[1].to(device)
            outputs = test_net(images_pre).to(device)  # 数据传入神经网络，计算输出
            predicted = torch.max(outputs.data, 1)[1]  # 获取最大能量的索引
            total += labels_pre.size(0)  # 计算预测次数
            correct += (predicted == labels_pre).sum().item()  # 计算正确预测次数

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # CIFAR-10中的分类
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # 生成两个dict,分别用来存放预测正确数量和总数量的个数
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # 启动预测过程，无需计算梯度等
    with torch.no_grad():
        for data in testloader:
            images_cl, labels_cl = data[0].to(device), data[1].to(device)
            outputs = net(images_cl).to(device)
            predictions = torch.max(outputs, 1)[1].to(device)
            # 开始计数
            for label, prediction in zip(labels_cl, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # 分类别打印预测准确率
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
