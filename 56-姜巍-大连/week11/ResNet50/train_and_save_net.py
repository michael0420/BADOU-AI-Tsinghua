"""
部署模型至GPU，训练模型并保存
"""

import datasets_process as dp  # 导入自建datasets
from MyResNet import MyResNet50  # 导入自定义神经网络
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.optim as optim

train_transform = transforms.Compose([
    # 随机旋转图片(原文无此处理，可选)transforms.RandomHorizontalFlip(),
    # 将图片尺寸resize到短边256，长边不变
    transforms.Resize(256),
    # 基于中心位置裁剪成224 × 224
    transforms.CenterCrop((224, 224)),
    # 将图片转化为Tensor
    transforms.ToTensor(),
    # 归一化[-1,1]之间(原文没有)
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = dp.MyDataset('./data/catVSdog/train.txt', transform=train_transform)

batch_size = 128
trainloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
# 类别信息也是需要我们给定的
classes = ('cat', 'dog')  # 对应label=0，label=1

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检测是否可以在GPU上运行训练过程
    print(f"model will be trained on device: '{device}'")

    net = MyResNet50().to(device)  # 实例化神经网络，存入GPU

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)

    print(f"Now comes the training procedure, batch size is {batch_size}.")

    epoches = 20

    for epoch in range(epoches):  # 数据被遍历的次数

        running_loss = 0.0  # 每次遍历前重新初始化loss值
        # loss_record = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)  # 切分数据集

            optimizer.zero_grad()  # 梯度清零，避免上一个batch迭代的影响

            # 前向传递 + 反向传递 + 权重优化
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 输出日志
            running_loss += loss.item()  # Tensor.item()方法是将tensor的值转化成python number

            if i % 15 == 14:  # 每90个mini batches输出一次
                print(f'[epoch:{epoch + 1}, mini-batch:{i + 1:3d}] loss: {running_loss / 15:.5f}')
                running_loss = 0.0

    print('Finished Training !')

    PATH = './MyResNet50.pth'
    torch.save(net.state_dict(), PATH)  # 保存训练好的模型到指定路径
