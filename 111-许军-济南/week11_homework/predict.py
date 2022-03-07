# -- coding:utf-8 --
from week11_homework.AlexNet import AlexNet
import torch
import cv2
import numpy as np
if __name__ == '__main__':
    model = AlexNet()
    model = torch.load("first_model2.pth")
    img = cv2.imread("./test4.jpg")
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255
    img = cv2.resize(img,(224,224))
    img = img.reshape(3,224,224)
    img = torch.tensor(img,dtype=torch.float32)
    img = img.unsqueeze(0).cuda()
    result = model(img)
    label = torch.argmax(result)

    if label == 0:
        print("cat")
    else:
        print("dog")
