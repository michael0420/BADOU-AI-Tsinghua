"""
将AlexNet进行适当修改用来识别"猫狗大战"(脆弱的笔记本电脑装不下ImageNet数据集，显存也不够……555)
1. 由于标签类别由1000锐减至2，在实际特征上也一定会数量可观地减少。因此我大胆地将每层的卷积核数量减半。
2. 全链接层神经元数量由原来的4096-4096-1000，变成2048-128-2(最后变成2分类)
"""

import torch.nn as nn
import torch.nn.functional as F


class ModifiedAlexNet(nn.Module):
    """编写ModifiedAlexNet"""

    def __init__(self):
        """
        初始化网络层：5层卷积层 + 3层全链接层
        初始化其他功能：Local Response Normalization + max pooling + dropout
        """

        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=48,
            kernel_size=(11, 11),
            stride=(4, 4),
            padding=(2, 2)
        )

        self.conv2 = nn.Conv2d(
            in_channels=48,
            out_channels=128,
            kernel_size=(5, 5),
            stride=(1, 1),
            padding=(2, 2)
        )

        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.conv4 = nn.Conv2d(
            in_channels=192,
            out_channels=192,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.conv5 = nn.Conv2d(
            in_channels=192,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.mxpool = nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=(2, 2)
        )

        self.lrn = nn.LocalResponseNorm(
            size=5,
            alpha=0.0001,
            beta=0.75,
            k=2
        )

        self.fc1 = nn.Linear(
            in_features=6 * 6 * 128,
            out_features=2048
        )

        self.fc2 = nn.Linear(
            in_features=2048,
            out_features=128
        )

        self.fc3 = nn.Linear(
            in_features=128,
            out_features=2
        )

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        前馈函数
        :param x: 神经网络输入(B, C, H, W) = (batch_size, 3, 224, 224)
        :return: 初步分类结果
        """

        # print(f"输入，shape{x.shape}")
        x = self.mxpool(self.lrn(F.relu(self.conv1(x))))  # 输入 > conv1卷积 > ReLu激活 > LRN > maxpool最大池化
        # print(f"第一次池化后，shape{x.shape}")
        x = self.mxpool(self.lrn(F.relu(self.conv2(x))))  # > conv2卷积 > ReLu激活 > LRN > maxpool最大池化
        # print(f"第二次池化后，shape{x.shape}")
        x = F.relu(self.conv3(x))  # > conv3卷积 > ReLu激活
        # print(f"第3层卷积后，shape{x.shape}")
        x = F.relu(self.conv4(x))  # > conv4卷积 > ReLu激活
        # print(f"第4层卷积后，shape{x.shape}")
        x = self.mxpool(F.relu(self.conv5(x)))  # > conv5卷积 > ReLu激活 > maxpool最大池化
        # print(f"第三次池化后，shape{x.shape}")
        x = x.view(-1, 6 * 6 * 128)  # 扁平化成1维
        x = F.relu(self.dropout(self.fc1(x)))  # > fc1全链接 > droupout > ReLu激活
        x = F.relu(self.dropout(self.fc2(x)))  # > fc2全链接 > droupout > ReLu激活
        x = self.fc3(x)  # > fc3全链接
        return x
