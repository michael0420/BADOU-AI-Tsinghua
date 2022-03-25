#coding:utf-8

'''

'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):  #双星号（**）将参数以字典的形式导入:
        support_optim = {
            'SGD':optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optim[optimist]

    def train(self, train_loader, epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):  #sequence–一个序列、迭代器或其他支持迭代对象。start–下标起始位置。
                inputs, labels = data
                self.optimizer.zero_grad()  #optimizer.zero_grad()意思是把梯度置零，也就是把loss关于weight的导数变成0.
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()         #反向传播求梯度
                self.optimizer.step()   #更新所有参数
                #这个是为了算loss的平均值，loss是个标量，在pytorch里用item取出这个唯一的元素
                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' % (epoch + 1, (i+1)*1./len(train_loader),running_loss / 100))
                    running_loss = 0.0
        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating...')
        correct = 0
        total = 0
        with torch.no_grad():   #no grad when test and predict
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)    #我们想要求每一行最大的列标号，我们就要指定dim=1，表示我们不要列了，保留行的size就可以了。假如我们想求每一列的最大行标，就可以指定dim=0，表示我们不要行了
                total += labels.size(0)     #将数组size第一个元素的值赋给变量x=size(0)/与tensor.size(0)是一样的，返回tensor形状的第一维数，一般是batch_size
                correct += (predicted == labels).sum().item()   #取出单元素张量的元素值并返回该值，保持原元素类型不变.精度更高
        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

#加载数据集
def mnist_load_data():
    transform = transforms.Compose(        #Compose把多个步骤整合到一起
        [transforms.ToTensor(),
        transforms.Normalize([0,], [1,])])

    #https://blog.csdn.net/ftimes/article/details/105202039
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=True, num_workers=2)
    return trainloader, testloader


class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)