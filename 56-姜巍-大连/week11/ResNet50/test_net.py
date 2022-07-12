"""
测试训练好的模型(用训练好的模型进行预测)
"""

from MyResNet import MyResNet50
import datasets_process as dp
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

test_transform = transforms.Compose([
    # 将图片尺寸resize到短边256
    transforms.Resize(256),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_data = dp.MyDataset('./data/catVSdog/test.txt', transform=test_transform)

batch_size = 64
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
# 类别信息也是需要我们给定的
classes = ('cat', 'dog')  # 对应label=0，label=1


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 检测是否可以在GPU上运行测试过程
    print(f"model will be tested on device: '{device}'")

    PATH = './Modified_Alex_net.pth'

    test_net = MyResNet50().to(device)
    test_net.load_state_dict(torch.load(PATH))
    test_net.eval()  # 测试模式eval()自动将batch-normalization、dropout等设置为False

    correct = 0
    total = 0
    # 由于这不是在训练模型，因此对输出不需要计算梯度等反向传播过程
    with torch.no_grad():
        for data in testloader:
            images_pre, labels_pre = data[0].to(device), data[1].to(device)
            outputs = test_net(images_pre)  # 数据传入神经网络，计算输出
            predicted = torch.max(outputs.data, 1)[1]  # 获取最大能量的索引
            total += labels_pre.size(0)  # 计算预测次数
            correct += (predicted == labels_pre).sum().item()  # 计算正确预测次数

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
