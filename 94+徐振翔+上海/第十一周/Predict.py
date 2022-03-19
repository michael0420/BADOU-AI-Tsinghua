import cv2
import numpy as np
from model.AlexNet import AlexNet
from model.Vgg16 import Vgg16Net
from model.Resnet50 import ResNet50
import utils

if __name__ == '__main__':
    # output_base_path = "D://dataset/AlexNet/AlexNet-Keras-master"
    # 加载需要预测的图像 (h,w,c)
    img = cv2.imread("./test_img/test.jpg")
    # print(img.shape)
    # 归一化 (h,w,c)
    img_nor = img / 255
    # print(img_nor.shape)
    # 增加一个维度，从(h,w,c)变为(n,h,w,c) (1, h, w, 3)
    img_resize = np.expand_dims(img_nor, axis=0)
    # 调整图像尺寸为 1*224*224*3 (n,h,w,c)
    img_resize = utils.resize(img_resize, (224, 224))
    # img_resize = np.resize(img_nor, (224, 224, 3))
    # print(img_resize.shape)
    # img_resize = cv2.resize(img_nor, (224, 224))
    # print(img_resize.shape)
    # 转换图像维度从(n,h,w,c)变为(n,c,h,w) (1, 3, 224, 224)
    # img_resize = img_resize.transpose(0, 3, 1, 2)
    # print(img_resize.shape)
    # 初始化AlexNet对象
    # 实例化模型
    if utils.net == 1:
        model = AlexNet()
    elif utils.net == 2:
        model = Vgg16Net()
    elif utils.net == 3:
        model = ResNet50()

    # 加载训练后的权重
    weights_path = utils.base_path + '/logs/last1.h5'
    # print(weights_path)
    model.load_weights(weights_path)
    # 预测
    predict = model.predict(img_resize)
    # 最大值索引
    argmax = np.argmax(predict)
    # 输出中文标签
    label = utils.print_answer(argmax)
    print("预测结果为:", label)
    cv2.imshow("predict", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
