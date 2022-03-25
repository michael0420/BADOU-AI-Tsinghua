from torch import nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=(11, 11), stride=(4, 4))
        self.batchNorm1 = nn.BatchNorm2d(96)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv2 = nn.Conv2d(96, 256, kernel_size=(5, 5))
        self.batchNorm2 = nn.BatchNorm2d(256)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = nn.Conv2d(256, 384, kernel_size=(3, 3))
        self.batchNorm3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=(3, 3))
        self.batchNorm4 = nn.BatchNorm2d(384)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=(3, 3))
        self.batchNorm5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(256 * 5 * 5, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = self.max_pool1(x)

        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = self.max_pool2(x)

        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        x = F.relu(self.batchNorm5(self.conv5(x)))

        x = x.view(-1, 256 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)
        x = self.fc3(x)
        return x
