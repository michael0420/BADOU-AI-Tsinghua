from torch import nn
import torch.nn.functional as F


# VGG 与AlexNet相比,加深了网络的深度,大卷积核用两个小卷积核代替
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        num = [16, 32, 64, 128, 128]
        self.conv1_1 = nn.Conv2d(3, num[0], kernel_size=(3, 3), padding=1)
        self.conv1_2 = nn.Conv2d(num[0], num[0], kernel_size=(3, 3), padding=1)

        self.conv2_1 = nn.Conv2d(num[0], num[1], kernel_size=(3, 3), padding=1)
        self.conv2_2 = nn.Conv2d(num[1], num[1], kernel_size=(3, 3), padding=1)

        self.conv3_1 = nn.Conv2d(num[1], num[2], kernel_size=(3, 3), padding=1)
        self.conv3_2 = nn.Conv2d(num[2], num[2], kernel_size=(3, 3), padding=1)
        self.conv3_3 = nn.Conv2d(num[2], num[2], kernel_size=(3, 3), padding=1)

        self.conv4_1 = nn.Conv2d(num[2], num[3], kernel_size=(3, 3), padding=1)
        self.conv4_2 = nn.Conv2d(num[3], num[3], kernel_size=(3, 3), padding=1)
        self.conv4_3 = nn.Conv2d(num[3], num[3], kernel_size=(3, 3), padding=1)

        self.conv5_1 = nn.Conv2d(num[3], num[4], kernel_size=(3, 3), padding=1)
        self.conv5_2 = nn.Conv2d(num[4], num[4], kernel_size=(3, 3), padding=1)
        self.conv5_3 = nn.Conv2d(num[4], num[4], kernel_size=(3, 3), padding=1)

        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.fc1 = nn.Linear(num[2] * 28 * 28, 4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.max_pool(x)

        # 由于数据集比较小,这里为了准确度,减少了模型层数
        # x = F.relu(self.conv4_1(x))
        # x = F.relu(self.conv4_2(x))
        # x = F.relu(self.conv4_3(x))
        # x = self.max_pool(x)

        # x = F.relu(self.conv5_1(x))
        # x = F.relu(self.conv5_2(x))
        # x = F.relu(self.conv5_3(x))
        # x = self.max_pool(x)
        # x = nn.Flatten(x)

        # VGG 模型本来就挺深了,硬件资源有限,不加dropout了
        # x = torch.flatten(x, start_dim=1)
        x = x.view(-1, 64 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
