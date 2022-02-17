import torch
import torchvision
# 涉及到的包：torch.nn, torch.optim, torchvision.transform

class torchMinist:
    def __init__(self, net, loss, optim):
        self.net = net
        self.loss = self.computeLoss(loss)
        self.optim = self.generateOptim(optim)

    # 计算交叉熵损失值和优化器
    def computeLoss(self, loss):
        support_cost = {
            "CROSS_ENTROPY": torch.nn.CrossEntropyLoss(),
            "MSE": torch.nn.MSELoss()
        }

        return support_cost[loss]

    def generateOptim(self, optim, **rests):  # **rests不明觉厉
        support_optim = {
            "SGD": torch.optim.SGD(self.net.parameters(), lr=0.1, **rests),
            "ADAM": torch.optim.Adam(self.net.parameters(), lr=0.01, **rests),
            "RMSP": torch.optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optim]

    # 定义训练过程和检验过程
    def train(self, train_loader, epoches=5):
        for epoch in range(epoches):
            zero_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                self.optim.zero_grad()  # ???

                outputs = self.net(inputs)
                loss2 = self.loss(outputs, labels)
                loss2.backward()
                self.optim.step()

                zero_loss += loss2.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss:%.3f' %
                          (epoch + 1, (i + 1) * 1. / len(train_loader), zero_loss / 100)
                          )
                    zero_loss = 0.0
        print("Train finished")

    def test(self, test_Loader):
        print("testing......")
        correct = 0
        total = 0
        with torch.no_grad():  # 不明觉厉
            for data in test_Loader:
                images, labels = data
                outputs = self.net(images)
                predict = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        print("the accuracy of tested images is %d %%" % (100 * correct / total))  # 两个百分号外接一个百分号，不明觉厉


def ministload():
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0,], [1,])])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=2)

    return trainloader, testloader


class ministNet(torch.nn.Module):
    def __init__(self):
        super(ministNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.softmax(self.fc3(x), dim=1)
        return x


if __name__ == '__main__':
    net = ministNet()
    model = torchMinist(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = ministload()
    model.train(train_loader)
    model.test(test_loader)
