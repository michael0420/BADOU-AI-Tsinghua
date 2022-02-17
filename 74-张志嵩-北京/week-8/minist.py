# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 10:03:11 2022

@author: Administrator
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
# import torchvision
from torch.autograd import Variable
# import numpy as np
# import matplotlib.pyplot as plt
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.softmax(self.fc2(x))
        return x


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 16, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU()
                                   )
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU())
        self.maxpool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(7 * 7 * 32, 100)
        self.fc2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)
        self.classifier = nn.Sequential(nn.Conv2d(32, 10, 3, 1, 1),
                                        nn.AdaptiveAvgPool2d(1)
                                        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        # x = self.fc1(x.view(x.size(0),-1))
        # x = self.fc2(x)
        # x = self.softmax(x)
        x = self.classifier(x)
        x = self.softmax(x.view(x.size(0), -1))
        return x


def main():
    epochs = 100
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [1])
                                    ])
    data_train = datasets.MNIST(root="./dataset", transform=transform, train=True,
                                download=True
                                )
    data_test = datasets.MNIST(root='./dataset', transform=transform, train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=128,
                                                    shuffle=True
                                                    )
    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=128,
                                                   shuffle=False
                                                   )

    model = CnnNet()  # Model()
    print(model)
    if torch.cuda.is_available():
        model.cuda()
    cost = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-1)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(data_loader_train),
                                              epochs=epochs)

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        print("Epoch{}/{}".format(epoch, epochs))
        print("-" * 10)
        # for i,data in enumerate(data_loader_train):
        pbar = tqdm(data_loader_train)
        for data in pbar:
            x, y = data
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            outputs = model(x)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.data.cpu().numpy()
            running_correct += torch.sum(pred == y)
            pbar.set_description(
                "Processing loss is %.4f, correct is %d" % (loss.data.cpu().numpy(), torch.sum(pred == y)))
        testing_correct = 0
        for i, data in enumerate(data_loader_test):
            x, y = data
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            #x, y = Variable(x), Variable(y)   #abandon after 0.4.0
            outputs = model(x)
            _, pred = torch.max(outputs, 1)
            testing_correct += torch.sum(pred == y)
        print("\n")
        print("Loss is :{:.5f}, Train Accuracy is {:.4f}%, Test Accuracy is: {:.4f}%, lr is: {:.4f}".format(
            running_loss / len(data_train), 100 * running_correct.data.cpu().numpy() / len(data_train),
            100 * testing_correct.data.cpu().numpy() / len(data_test), optimizer.state_dict()['param_groups'][0]['lr']))

if __name__ == "__main__":
    main()