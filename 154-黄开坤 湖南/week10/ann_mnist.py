#coding:utf-8

'''
keras框架下实现MNIAST数据集识别
模型：ANN
https://www.cnblogs.com/ncuhwxiong/p/9774515.html
https://blog.csdn.net/weixin_42886817/article/details/99831718
'''

import numpy as np
from keras.datasets import mnist
from keras.models import Sequential,Model   #顺序模型，类模型
from keras.layers import Dense, Dropout
from keras.utils import np_utils

seed = 7    # 设置随机数种子
np.random.seed(seed)

'''
数据集是3维的向量（instance length,width,height).对于多层感知机，模型的输入是二维的向量，因此这
里需要将数据集reshape，即将28*28的向量转成784长度的数组。可以用numpy的reshape函数轻松实现这个过程。
'''
# 数据处理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
num_pixels = x_train.shape[1] * x_train.shape[2]    # 像素个数
x_train= x_train.reshape(x_train.shape[0], num_pixels).astype('float32')    # reshape+转换数据类型
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')
# 归一化
x_train = x_train / 255
x_test = x_test / 255
'''
最后，模型的输出是对每个类别的打分预测，对于分类结果从0-9的每个类别都有一个预测分值，表示将模型
输入预测为该类的概率大小，概率越大可信度越高。由于原始的数据标签是0-9的整数值，
通常将其表示成0ne-hot向量。如第一个训练数据的标签为5，one-hot表示为[0,0,0,0,0,1,0,0,0,0]。
'''
y_train = np_utils.to_categorical(y_train)  #to_categorical就是将类别向量转换为二进制（只有0和1）的矩阵类型表示。其表现为将原有的类别向量转换为独热编码的形式。
y_test = np_utils.to_categorical(y_test)    #shape=(1000, 10)
num_classes = y_test.shape[1]   # 矩阵的列

# 现在需要做得就是搭建神经网络模型了，创建一个函数，建立含有一个隐层的神经网络。
def baseline_model():
    model = Sequential()    #Sequential 模型结构： 层（layers）的线性堆栈。简单来说，它是一个简单的线性结构，没有多余分支，是多个网络层的堆叠。
    # 中间层节点数，输入形状=input_shape(784, )，权值矩阵初始化，激活
    model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))  #隐层
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  #编译
    return model
'''
型的隐含层含有784个节点，接受的输入长度也是784（28*28），最后用softmax函数将预测结果转换为标签的概率值。 
将训练数据fit到模型，设置了迭代轮数，每轮200个训练样本，将测试集作为验证集，并查看训练的效果。
'''
model = baseline_model()
# fit the model。https://www.freesion.com/article/7674702178/
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=2)
# evaluate。https://blog.csdn.net/qq_28979491/article/details/101529849
scores = model.evaluate(x_test, y_test, verbose=0)  #返：loss,accuracy
print("Baseline Error: %.2f%%" % (100-scores[1]*100))