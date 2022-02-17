#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : bxh
# @Time     : 2022/1/22 11:20
# @File     : keras.py
# @Project  : 神经网络实现

from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

(tran_images,train_labels),(test_images,test_labels) = mnist.load_data()
print(tran_images.shape)
print(test_images.shape)
print(train_labels)
print(test_labels)

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
tran_images = tran_images.reshape((60000,28*28))
tran_images = tran_images.astype('float')/255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float')/255
#对标签one_hot处理
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(train_labels)

network.fit(tran_images,train_labels,epochs=10,batch_size=80)
test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=1)
print('test_loss = ',test_loss)
print('test_acc = ',test_acc)