import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

"""
实现模型的训练和预测
"""
class Model:
    """
    net 定义的神经网络
    cost 损失函数
    optimist 优化算法
    """
    def __init__(self,net,cost,optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass
    """
    根据输入的cost名称，匹配torch的损失函数，
    支持交叉熵和MSE损失函数
    """
    def create_cost(self,cost):
        support_cost = {
            "CROSS_ENTROPY":nn.CrossEntropyLoss(),
            "MSE":nn.MSELoss()
        }
        return support_cost[cost]
    """
    根据输入的优化函数名，匹配哟话函数
    支持SGD，RMSProp，Adam
    """
    def create_optimizer(self,optimist,**rests):
        support_optim = {
            "SGD":optim.SGD(self.net.parameters(),lr=0.1, **rests),
            "ADAM":optim.Adam(self.net.parameters(),lr=0.01, **rests),
            "RMSP":optim.RMSprop(self.net.parameters(),lr=0.001, **rests)
        }
        return support_optim[optimist]

    """
    训练函数
    输入训练数据集和迭代次数
    """
    def train(self,train_loader,epoches=3):
        for epoch in range(epoches):
            running_loss = 0.0
            #遍历训练集中的数据
            for i,data in enumerate(train_loader,0):
                inputs,lables = data
                #优化器梯度初始化为0，？？？
                self.optimizer.zero_grad()
                #forward+backword+optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs,lables)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print("Finished Training")

    """
    执行预测集，评估模型
    """
    def evaluate(self,test_loader):
        print("Evaluating ...")
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images,labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs,1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))


def mnist_load_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0,],[1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)
    return trainloader, testloader

"""
定义神经网络结构
该神经网络需要集成torch.nn.Model,并在初始化函数中创建网络需要包含的层，并实现forward函数完成前向计算
网络的反向计算由自动求导机制处理
通常需要训练的层卸载init函数中，将参数不需要训练的层在forward方法里调用对应的函数来实现相应的层
"""
class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet,self).__init__()
        #输入层
        self.fc1 = torch.nn.Linear(28*28,512)
        #隐层
        self.fc2 = torch.nn.Linear(512,512)
        #输出层
        self.fc3 = torch.nn.Linear(512,10)
    #前向计算
    def forward(self,x):
        #输入reshape成28*28的张量
        x = x.view(-1,28*28)
        #输入层，经过relu的激活函数
        x = F.relu_(self.fc1(x))
        # 隐层，经过relu的激活函数
        x = F.relu_(self.fc2(x))
        #输出层，经过softmax
        x = F.softmax(self.fc3(x),dim=1)
        return x

if __name__ == '__main__':
    #初始化神经网络
    net = MnistNet()
    model = Model(net,"CROSS_ENTROPY","RMSP")
    train_loader,test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)





