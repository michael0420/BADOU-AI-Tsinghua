# -- coding:utf-8 --
import cv2
import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from week11_homework import AlexNet
from week11_homework import ResNet
from week12_homework import InceptionV3
from week12_homework import MobileNet
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
class AnaDataset(torch.utils.data.Dataset):
        def __init__(self, data_dir, transform=None) :

            self.label_name={"cat" : 0, "dog" : 1}
            self.data_info=self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
            self.transform=transform

        def __getitem__(self, index) :
            path_img, label=self.data_info[index]

            if self.transform is not None :
                path_img=self.transform(path_img)  # 在这里做transform，转为tensor等等

            return path_img, label

        def __len__(self) :
            return len(self.data_info)

        @staticmethod
        def get_img_info(data_dir) :
            data_info=list()
            # 获取总长度
            n=len(data_dir)
            for i in range (n):
                name=lines[i].split(';')[0]
                # 从文件中读取图像
                img=cv2.imread(r".\data\image\train" + '/' + name)
                img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img,(224,224))
                img = img.reshape(3,224,224)
                img = img.astype(np.float32)/255
                label = int(lines[i].split(';')[1].strip("\n"))
                data_info.append((img,label))
            return data_info






if __name__ == '__main__':
    log_dir = "./logs/"
    with open(r"./data/dataset.txt","r") as f:
        lines = f.readlines()
    # 打乱数据
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    # 训练集和测试集为9:1
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val
    # 建立模型
    model = MobileNet.MobileNet()
    # if torch.cuda.is_available():
    # model=model.cuda()
    model.train(True)
    # 学习率的方式
    lr = 0.01
    # 损失函数
    # if torch.cuda.is_available():
    criterion = nn.CrossEntropyLoss().cuda()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),lr = lr)
    train_data = AnaDataset(lines[:num_train])
    # g = generate_arrays_from_file(lines[:num_train],batch_size)
    train_loader=DataLoader(dataset=train_data, batch_size=16, shuffle=True)
    # 开始训练
    for epoch in range(10):
        for i,data in enumerate(train_loader):
            # if torch.cuda.is_available():
            batch_img,batch_label = data
                # batch_img = batch_img.cuda()
                # batch_label = batch_label.cuda()
            optimizer.zero_grad()
            output = model(batch_img)
            loss = criterion(output,batch_label)
            loss.backward()
            optimizer.step()
        print(epoch)
    torch.save(model,"first_model3_resnet.pth")



