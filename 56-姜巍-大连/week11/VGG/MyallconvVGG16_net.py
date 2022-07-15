import torch
import torch.nn as nn
import torch.nn.functional as F


class MyVgg16(nn.Module):
    """写一个Vgg16/19全卷积神经网络"""

    def __init__(self):
        """
        继承nn.Module属性，
        初始化自己独有的属性: 13/16个卷积层、1种池化层、3个卷积替代全链接层
        """

        super().__init__()
        self.conv_1_1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.batch_norm_1_1 = nn.BatchNorm2d(64)

        self.conv_1_2 = nn.Conv2d(64, 64, 3, 1, 1)

        self.batch_norm_1_2 = nn.BatchNorm2d(64)

        self.conv_2_1 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.batch_norm_2_1 = nn.BatchNorm2d(128)

        self.conv_2_2 = nn.Conv2d(128, 128, 3, 1, 1)

        self.batch_norm_2_2 = nn.BatchNorm2d(128)

        self.conv_3_1 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.batch_norm_3_1 = nn.BatchNorm2d(256)

        self.conv_3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.batch_norm_3_2 = nn.BatchNorm2d(256)

        self.conv_3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.batch_norm_3_3 = nn.BatchNorm2d(256)

        self.conv_3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.batch_norm_3_4 = nn.BatchNorm2d(256)

        self.conv_4_1 = nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.batch_norm_4_1 = nn.BatchNorm2d(512)

        self.conv_4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm_4_2 = nn.BatchNorm2d(512)

        self.conv_4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm_4_3 = nn.BatchNorm2d(512)

        self.conv_4_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm_4_4 = nn.BatchNorm2d(512)

        self.conv_5_1 = nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1)
        )

        self.batch_norm_5_1 = nn.BatchNorm2d(512)

        self.conv_5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm_5_2 = nn.BatchNorm2d(512)

        self.conv_5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm_5_3 = nn.BatchNorm2d(512)

        self.conv_5_4 = nn.Conv2d(512, 512, 3, 1, 1)
        self.batch_norm_5_4 = nn.BatchNorm2d(512)

        self.mxpool = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        )

        self.conv_6_1 = nn.Conv2d(
            in_channels=512,
            out_channels=2048,
            kernel_size=(7, 7),
            stride=(1, 1)
        )

        self.conv_6_2 = nn.Conv2d(
            in_channels=2048,
            out_channels=128,
            kernel_size=(1, 1),
            stride=(1, 1)
        )

        self.conv_6_3 = nn.Conv2d(
            in_channels=128,
            out_channels=2,
            kernel_size=(1, 1),
            stride=(1, 1)
        )

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """前馈函数"""

        x = F.relu(self.batch_norm_1_1(self.conv_1_1(x)))
        x = self.mxpool(F.relu(self.batch_norm_1_2(self.conv_1_2(x))))
        x = F.relu(self.batch_norm_2_1(self.conv_2_1(x)))
        x = self.mxpool(F.relu(self.batch_norm_2_2(self.conv_2_2(x))))
        x = F.relu(self.batch_norm_3_1(self.conv_3_1(x)))
        x = F.relu(self.batch_norm_3_2(self.conv_3_2(x)))
        # x = F.relu(self.batch_norm_3_3(self.conv_3_3(x)))
        x = self.mxpool(F.relu(self.batch_norm_3_4(self.conv_3_4(x))))
        x = F.relu(self.batch_norm_4_1(self.conv_4_1(x)))
        x = F.relu(self.batch_norm_4_2(self.conv_4_2(x)))
        # x = F.relu(self.batch_norm_4_3(self.conv_4_3(x)))
        x = self.mxpool(F.relu(self.batch_norm_4_4(self.conv_4_4(x))))
        x = F.relu(self.batch_norm_5_1(self.conv_5_1(x)))
        x = F.relu(self.batch_norm_5_2(self.conv_5_2(x)))
        # x = F.relu(self.batch_norm_5_3(self.conv_5_3(x)))
        x = self.mxpool(F.relu(self.batch_norm_5_4(self.conv_5_4(x))))
        x = F.relu(self.dropout(self.conv_6_1(x)))
        x = F.relu(self.dropout(self.conv_6_2(x)))
        x = self.conv_6_3(x)
        x = torch.squeeze(x)  # 删除维度为1的维度

        return x
