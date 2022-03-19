# 输入输出接口

import cv2
import numpy as np
import tensorflow as tf

base_path = "D://dataset/AlexNet/AlexNet-Keras-master"
# net = 1  # AlexNet
# net = 2  # Vgg16
net = 3  # Resnet50

# 调整图片的尺寸
def resize(inputs, shape):
    # tf.name_scope()
    # 命名空间的实际作用
    # （1）在某个tf.name_scope()
    # 指定的区域中定义的所有对象及各种操作，他们的“name”属性上会增加该命名区的区域名，用以区别对象属于哪个区域；
    # （2）将不同的对象及操作放在由tf.name_scope()
    # 指定的区域中，便于在tensorboard中展示清晰的逻辑关系图，这点在复杂关系图中特别重要。
    # 类似于C++的namespace
    with tf.name_scope('resize_img'):
        images = []
        for img in inputs:
            img = cv2.resize(img, shape)
            images.append(img)
        # 将图片转为ndarray
        images = np.array(images)
        return images


def print_answer(argmax):
    # 输入分类数（0，1，2...），输出分类结果（猫，狗...）
    with open(base_path + '/data/model/index_word.txt', 'r', encoding='utf-8') as f:
        # 以分号分隔,strip：去除空格
        labels = [line.split(';')[1][:-1] for line in f.readlines()]
    print(labels)
    return labels[argmax]