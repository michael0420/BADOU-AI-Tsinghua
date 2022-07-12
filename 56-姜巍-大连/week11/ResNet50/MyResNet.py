"""
根据：
    ResNet论文《Deep Residual Learning for Image Recognition》；
    PyTorch官方的自定义MODULES教程https://pytorch.org/docs/stable/notes/modules.html
自定义Conv Block 和 Identity Block并作为"layer"添加进ResNet50的神经网络结构中
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """编写ConvBlock代码"""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, down_sampling=1):
        """
        初始化网络
        :param in_channels: Conv. Block 输入通道数
        :param hidden_channels: Conv. Block 中间通道数
        :param out_channels: Conv. Block 输出通道数
        :param down_sampling: 当需要down sampling时，请设置该参数为2(默认为1，即不进行down sampling)，赋值到 3×3 conv层的stride参数
        """

        super().__init__()

        # 左侧分支序贯网络
        self.conv_bn_relu_stack = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=3,
                stride=down_sampling,
                padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),

            nn.Conv2d(hidden_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels)
        )

        # 右侧分支序贯网络
        self.conv_bn_stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=down_sampling),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        """完成ConvBlock传递"""

        x = F.relu(self.conv_bn_stack(x) + self.conv_bn_relu_stack(x))
        return x


class IdentityBlock(nn.Module):
    """编写IdentityBlock代码"""

    def __init__(self, in_out_channels: int, mid_channels: int):
        """
        初始化属性
        :param in_out_channels: Identity Block 输入通道数
        :param mid_channels: Identity Block 输出通道数
        """

        super().__init__()

        self.conv_bn_relu_stack = nn.Sequential(
            nn.Conv2d(in_out_channels, mid_channels, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),

            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),

            nn.Conv2d(mid_channels, in_out_channels, 1, 1),
            nn.BatchNorm2d(in_out_channels)
        )

    def forward(self, x):
        """完成网络传递"""

        x = F.relu(x + self.conv_bn_relu_stack(x))
        return x


class MyResNet50(nn.Module):
    """编写ResNet50代码"""

    def __init__(self, input_channels=3):
        """
        初始化属性
        :param input_channels: ResNet50 输入图片通道数
        """

        super().__init__()

        self.ResNet50 = nn.Sequential(
            # Conv1:
            # NO. of feature maps: from 3 to 64
            # down sampling: from 224 to 112(by conv op.) to 56(by max-pooling op.)
            nn.Conv2d(input_channels, 64, (7, 7), (2, 2), (3, 3)),  # in, out, kernel, stride, padding
            nn.MaxPool2d(3, 2, 1),  # kernel, stride, padding

            # Conv2_x
            # NO. of feature maps: from 64 to 256
            # down sampling: None
            ConvBlock(64, 64, 256),
            IdentityBlock(256, 64),
            IdentityBlock(256, 64),

            # Conv3_x
            # NO. of feature maps: from 256 to 512
            # down sampling: from 56 to 28
            ConvBlock(256, 128, 512, down_sampling=2),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128),
            IdentityBlock(512, 128),

            # Conv4_x
            # NO. of feature maps: from 512 to 1024
            # down sampling: from 28 to 14
            ConvBlock(512, 256, 1024, down_sampling=2),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),
            IdentityBlock(1024, 256),

            # Conv5_x
            # NO. of feature maps: from 1024 to 2048
            # down sampling: from 14 to 7
            ConvBlock(1024, 512, 2048, down_sampling=2),
            IdentityBlock(2048, 512),
            IdentityBlock(2048, 512),

            # global average pool
            # down sampling: from 7 to 1
            nn.AvgPool2d(7),

            # 将c h w平铺
            nn.Flatten(),

            # 全链接层Liner
            # nn.Linear(2048, 1000)
            nn.Linear(2048, 2)  # 对猫狗二分类，所以对分类网络修改
        )

    def forward(self, x):
        """前馈函数，完成前向传递"""

        return self.ResNet50(x)
