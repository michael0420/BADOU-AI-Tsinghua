# 编写神经网络代码所需库
import torch.nn as nn
import torch.nn.functional as F


class AlexNetPyTorch(nn.Module):
    """编写AlexNet神经网络"""

    def __init__(self):
        """
        初始化网络层：5层卷积层 + 3层全链接层
        初始化其他功能：Local Response Normalization + max pooling + dropout
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=96,  # 原文由于当时单个GPU显存限制而将96个filter分成两个48filter分别在两个GPU上运行
            kernel_size=(11, 11),
            stride=(4, 4),
            padding=(2, 2)
        )

        self.conv2 = nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )

        self.conv3 = nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.conv4 = nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.conv5 = nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.mxpool = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2)
        )

        self.lrn1 = nn.LocalResponseNorm(
            size=5,
            alpha=0.0001,
            beta=0.75,
            k=2
        )

        self.lrn2 = nn.LocalResponseNorm(
            size=5,
            alpha=0.0001,
            beta=0.75,
            k=2
        )

        self.fc1 = nn.Linear(
            in_features=6 * 6 * 256,
            out_features=4096
        )

        self.fc2 = nn.Linear(
            in_features=4096,
            out_features=4096
        )

        self.fc3 = nn.Linear(
            in_features=4096,
            out_features=1000
        )

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        前馈函数
        :param x: 神经网络输入(B, C, H, W) = (batch_size, 3, 224, 224)
        :return: 初步分类结果
        """

        x = self.mxpool(self.lrn1(F.relu(self.conv1(x))))  # 输入 > conv1卷积 > ReLu激活 > LRN > maxpool最大池化
        x = self.mxpool(self.lrn2(F.relu(self.conv2(x))))  # > conv2卷积 > ReLu激活 > LRN > maxpool最大池化
        x = F.relu(self.conv3(x))  # > conv3卷积 > ReLu激活
        x = F.relu(self.conv4(x))  # > conv4卷积 > ReLu激活
        x = self.mxpool(F.relu(self.conv5(x)))  # > conv5卷积 > ReLu激活 > maxpool最大池化
        x = x.view(-1, 6 * 6 * 256)  # 扁平化成1维
        x = F.relu(self.dropout(self.fc1(x)))  # > fc1全链接 > droupout > ReLu激活
        x = F.relu(self.dropout(self.fc2(x)))  # > fc2全链接 > droupout > ReLu激活
        x = self.fc3(x)  # > fc3全链接
        return x
