#!/usr/bin/env python 
# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers
from keras.datasets import mnist
from keras.utils import to_categorical

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 搭建神经网络,设置输入,输出节点,以及激活函数
network = models.Sequential()
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))
# 编译,设置优化器,损失函数,度量
network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
# 将数据归一化
train_images = np.asfarray(train_images.reshape(60000, 28 * 28)) / 255 * 0.99 + 0.01
test_images = np.asfarray(test_images.reshape(10000, 28 * 28)) / 255 * 0.99 + 0.01
# 数据标签改为独热编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# 训练
network.fit(train_images, train_labels, batch_size=256, epochs=5)
# 评估
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test_loss:\n", test_loss)
print("test_acc:\n", test_acc)

# 输入一个图片,进行验证
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
first_pic = test_images[0]
plt.imshow(first_pic, cmap="gist_yarg")
# plt.show()
# 归一化
test_images = np.asfarray(test_images.reshape(10000, 28 * 28)) / 255 * 0.99 + 0.01
# 预测
result = network.predict(test_images)
# 预测结果的第一张图,最大值的索引
max_value_idx = np.argmax(result[0])
print("图片对应的数字是", max_value_idx)
