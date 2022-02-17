import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,10)

    def forward(self, input_data):
        x = input_data.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def caltestaccuracy(model, testloader):
    correct = 0
    size=len(testloader.dataset)
    for i,batch_data in enumerate(testloader):
        input,labels = batch_data
        output_data = model(input)
        batch_correct = (output_data.argmax(1) == labels).type(torch.float).sum().item()
        correct += batch_correct
    correct /= size
    return correct



if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0,1)])
    trainset = torchvision.datasets.MNIST("./data",True,transform,download=True)
    testset = torchvision.datasets.MNIST("./data",True,transform,download=True)
    batch_size=128
    trainloader = DataLoader(trainset,batch_size,True,num_workers=2)
    testloader = DataLoader(testset,batch_size,False,num_workers=2)
    model = MnistNet()
    if os.path.exists("net.pth"):
        model.load_state_dict(torch.load("net.pth"))
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    lossfun = nn.CrossEntropyLoss()
    epoches = 10
    train_loss = []
    for epoch in range(epoches):
        model.train()
        running_loss = 0.0
        trainaccuracy = 0.0
        size = len(trainloader.dataset)
        batch_len = len(trainloader)
        for i, batch_data in enumerate(trainloader):
            input_data, labels = batch_data
            output_data = model(input_data)
            loss = lossfun(output_data,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.save(model.state_dict(), "net.pth")
            running_loss += loss.item()
            with torch.no_grad():
                batch_accuracy = (output_data.argmax(1) == labels).type(torch.float).sum().item()
                trainaccuracy += batch_accuracy
                batch_accuracy /= batch_size
                running_loss += loss.item()
                # if(i%200==0):
                #     print("epoch=%d,percent=%.2f,batch_loss=%.2f,batch_accuracy=%.2f"
                #           % (epoch, round((i+1)/batch_len,2), loss.item(), batch_accuracy))
        trainaccuracy /= size
        running_loss /= batch_len
        testaccuracy = caltestaccuracy(model, testloader)
        train_loss.append(running_loss)
        print("epoch=%d,loss=%.2f,trainacc=%.2f,testacc=%.2f" % (epoch, running_loss,trainaccuracy,testaccuracy))

    testloader = DataLoader(testset, 16, True)
    imgs,labels = next(iter(testloader))
    test_output = model(imgs)
    test_label = test_output.argmax(1)
    # 损失
    plt.figure()
    plt.plot(train_loss)
    # 预测
    plt.figure()
    for i, imgdata in enumerate(imgs):
        img = np.squeeze(imgdata)
        img = img.numpy()
        plt.subplot(4,4,i+1)
        plt.subplots_adjust(hspace=0.5)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        plt.xlabel("target:%d" %(labels[i].item()))
        plt.ylabel("predict:%d" %(test_label[i].item()))
    plt.show()















































