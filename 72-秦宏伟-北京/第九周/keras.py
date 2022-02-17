# -*- coding: utf-8 -*-
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import numpy as np
#准备数据
#####################################################################
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)
#60000行，28*28个元素的数据集
train_images = train_images.reshape((60000, 28*28))
#归一化
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

#onehot转换，例如将7转换为[0,0,0,0,0,0,0,1,0,0,]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#####################################################################

#构建模型
#####################################################################
network = models.Sequential()
#添加全连接隐藏层
network.add(layers.Dense(200,activation='sigmoid',input_shape=(28*28,)))
#添加全连接输出层
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])
#####################################################################

#训练模型
#####################################################################
network.fit(train_images,train_labels,epochs=10,batch_size=100)
#####################################################################

#评估模型(基于测试集)
#####################################################################
#verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=0)
print(test_loss)
print('test_acc', test_acc)
#####################################################################

test_data_file = open("mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (np.asfarray(all_values[1:]))
    inputs = inputs.reshape((1, 28 * 28))
    inputs = inputs.astype('float32') / 255
    #让网络判断图片对应的数字
    res = network.predict(inputs)
    label = np.argmax(res)
    print("网络认为图片的数字是：", label)
