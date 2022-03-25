import torch
from torch.autograd import Variable
from torch import nn
from torch import optim

from nets_config import nets
from data_loader import data_loader

# 这里可以选择 PALM-Training400 和 PALM-Training1200
# AlexNet可以使用PALM-Training400, 其他网络建议使用PALM-Training1200,否则数据量太少了,准确度上不去
train_dir = "../PALM-Training1200"

# 训练轮次
EPOCH_NUM = 20

criterion = nn.CrossEntropyLoss()


def train(model, optimizer, save_path):
    for epoch in range(EPOCH_NUM):
        num = 0
        for imgs, labels in data_loader(train_dir):
            num += len(imgs)
            imgs, labels = torch.Tensor(imgs), torch.Tensor(labels).long()
            imgs, labels = Variable(imgs), Variable(labels)
            optimizer.zero_grad()
            output = model(imgs)
            if isinstance(output, tuple):  # AlexNet VGG只有一个输出,GoogLeNet模型有三个输出分支
                output = output[0] + 0.3 * output[1] + 0.3 * output[2]
            # loss 用交叉熵,这里有个坑,pytorch的CrossEntropyLoss里面自带了softmax,所以网络最后一层不要加softmax
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_num = loss.item()
            print(f"epoch:{epoch}  {num}/400  loss:{loss_num}")
    torch.save(model.state_dict(), save_path)
    print(f"模型保存至 {save_path}")


if __name__ == '__main__':
    # 启动时要选一个模型,默认是AlexNet,
    # 可以修改为 VGG  GoogLeNet  ResNet  MobileNet
    net = nets["AlexNet"]

    model = net.get("model")
    path = net.get("path")
    # 这里可以尝试使用不同的优化器
    optimizer = optim.Adam(model.parameters())
    train(model, optimizer, path)
