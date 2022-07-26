import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable

cuda = torch.cuda.is_available()


def getDataLoader():
    transform = transforms.Compose([
        transforms.ToTensor(),  # range[0,255] -> [0.0, 1.0]
        transforms.Normalize(mean=(0.5), std=(0.5))  # (channel-mean)/std  range[0.0,1.0] -> [-0.5,0.5]
    ])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    # trainset 60000  testset 10000
    batch_size = 100
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
    return trainloader, testloader


# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 从28*28*1 开始
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        # 24*24*6
        self.pool1 = nn.MaxPool2d(2, 2)
        # 12*12*6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        # 8*8*16
        self.pool2 = nn.MaxPool2d(2, 2)
        # 4*4*16

        # 全连接
        self.fc1 = nn.Linear(4 * 4 * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 4 * 4 * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 创建网络
net = Net()
if cuda:
    net = Net.cuda()


def train(trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(5):
        for i, (imgs, labels) in enumerate(trainloader):
            if cuda:
                imgs, labels = imgs.cuda(), labels.cuda()
            imgs, labels = Variable(imgs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 50 == 0:
                print("epoch:{}  index:{}  loss:{}".format(epoch + 1, i + 1, loss))


def predict(testloader):
    rightNum = 0
    for i, (imgs, labels) in enumerate(testloader):
        if cuda:
            imgs, labels = imgs.cuda(), labels.cuda()
        output = net(Variable(imgs))
        indexs = torch.argmax(output, dim=1)
        for i in range(len(labels)):
            if labels[i].item() == indexs[i].item():
                rightNum += 1
    print("测试集10000张图,其中预测正确: {}".format(rightNum))  # 正确率 90%以上


if __name__ == '__main__':
    trainloader, testloader = getDataLoader()
    train(trainloader)
    predict(testloader)
