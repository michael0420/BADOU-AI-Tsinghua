from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from MyResNet import MyResNet50

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MyResNet50().to(device)
    model.load_state_dict(torch.load('./MyResNet50.pth'))  # 加载模型
    model.eval()  # 把模型转为test模式

    # 读取要预测的图片
    # 读取要预测的图片
    img = Image.open("./test_cat.jpg")  # 读取图像
    # img.show()
    plt.imshow(img)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img = trans(img)
    img = img.to(device)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]

    # 预测
    classes = ('cat', 'dog')
    output = model(img)
    prob = F.softmax(output, dim=1)  # prob是2个分类的概率
    print("概率：", prob)
    value, predicted = torch.max(output.data, 1)
    predict = output.argmax(dim=1)
    pred_class = classes[predicted.item()]
    print("预测类别：", pred_class)
