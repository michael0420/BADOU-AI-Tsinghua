import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


def data_loader(batch_size=10, shuffle=True):
    transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0], [1])])

    train_set = torchvision.datasets.MNIST(root="./dataset", train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    test_set = torchvision.datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return train_loader, test_loader


def conv_block(in_channels=3, out_channels=3, stride=(1, 1), pad=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=(3, 3), stride=stride, padding=pad),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class MnistNet(nn.Module):
    def __init__(self, class_num=10):
        super(MnistNet, self).__init__()
        self.class_num = class_num
        self.feature = nn.Sequential(
            conv_block(1, 3),
            conv_block(3, 3),
            conv_block(3, 3),
            conv_block(3, 3),
            conv_block(3, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(784, 512, bias=False),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 512, bias=False),
            # nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 10, bias=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y


class Model:
    def __init__(self, net, batch_size, epochs=4, lr=0.1):
        self.net = net
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=lr)
        self.cost = torch.nn.CrossEntropyLoss()
        self.epochs = epochs

    def train(self, train_loader):
        for epoch in range(self.epochs):
            for i, data in enumerate(train_loader):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.net(inputs)


                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if i % 100 == 0:
                    predicts = torch.argmax(outputs, dim=1)
                    accuracy = (predicts == labels).sum().item() / len(labels) * 100
                    print("epoch: %d, batch: %.3f%%, loss: %.3f, accuracy: %.3f%%"
                          % (epoch+1, i*100 / len(train_loader), loss.item(), accuracy))

    def evaluate(self, test_loader):
        with torch.no_grad():
            num_correct = 0
            num_sample = 0
            for i, data in enumerate(test_loader):
                inputs, labels = data
                outputs = self.net(inputs)
                predicts = torch.argmax(outputs, 1)
                num_correct += (predicts == labels).sum().item()
                num_sample += len(labels)
                accuracy = (predicts == labels).sum().item() / len(labels) * 100

                print("accuracy: %.2f%%" % accuracy)

            print ("total accuracy: %.2f%%" % (num_correct*100/num_sample))


if __name__ == '__main__':
    batch_size = 32
    epochs = 3
    lr = 0.001
    train_loader, test_loader = data_loader(batch_size)
    net = MnistNet()
    model = Model(net, batch_size, epochs, lr)
    model.train(train_loader)
    model.evaluate(test_loader)
