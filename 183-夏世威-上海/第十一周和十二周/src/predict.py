import torch
from torch.autograd import Variable

from nets_config import nets
from data_loader import valid_data_loader

valid_dir = "../PALM-Validation400"
csv_path = "../labels.csv"


def predict(model, save_path):
    model.load_state_dict(torch.load(save_path))
    actual_num = 0
    all_num = 0
    for imgs, labels in valid_data_loader(valid_dir, csv_path):
        imgs = torch.tensor(imgs)
        result = model(Variable(imgs))
        if isinstance(result, tuple):
            result = result[0]  # AlexNet VGG只有一个输出,GoogLeNet模型有三个输出分支
        result = torch.argmax(result, dim=1)
        for i in range(len(labels)):
            if int(labels[i]) == int(result[i]):
                actual_num += 1
            all_num += 1  # 这个最后应该是400,测试集 验证集 各400
        print("预测中 {}/400     目前正确率:{}%".format(all_num, actual_num / all_num * 100.0))
    print("最终模型正确率:{}%".format(actual_num / all_num * 100.0))


if __name__ == '__main__':
    # 启动时要选一个模型,默认是AlexNet
    # 可以修改为 VGG  GoogLeNet  ResNet  MobileNet
    net = nets["AlexNet"]

    model = net.get("model")
    path = net.get("path")
    predict(model, path)
