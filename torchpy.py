import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import cv2
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

train_dataset = torchvision.datasets.MNIST(
    root='data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root='data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)


batch_size = 64
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True
)

images, labels = next(iter(train_loader))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1,2,0)#变换为(size,channels)
print(labels)
cv2.imshow('win',img)
cv2.waitKey(0)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1,6,3,1,2),nn.ReLU(),nn.MaxPool2d(2,2))
        self.conv2=nn.Sequential(nn.Conv2d(6,16,5),nn.ReLU(),nn.MaxPool2d(2,2))
        self.fc1=nn.Sequential(nn.Linear(16*5*5,120),nn.BatchNorm1d(120),nn.ReLU())
        self.fc2=nn.Sequential(nn.Linear(120,84),nn.BatchNorm1d(84),nn.ReLU(),nn.Linear(84,10))
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size()[0],-1)
        x=self.fc1(x)
        x=self.fc2(x)
        return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.001

net = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    net.parameters(),
    lr=LR
)

epoch = 1
if __name__ == '__main__':
    for epoch in range(epoch):
        sum_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d,%d] loss: %.03f' % (epoch+1, i+1,sum_loss/100))
                sum_loss = 0.0

    net.eval()
    correct = 0
    total = 0
    for data_test in test_loader:
        images,labels=data_test
        images, labels=Variable(images), Variable(labels)
        output_test = net(images)
        _,predicted = torch.max(output_test,1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('correct1:', correct)
    print('test acc:{0}'.format(correct.item()/len(test_dataset)))



