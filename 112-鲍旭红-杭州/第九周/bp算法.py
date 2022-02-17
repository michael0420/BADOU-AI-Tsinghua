#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : bxh
# @Time     : 2022/1/22 7:55
# @File     : bp算法.py
# @Project  : 神经网络实现

import numpy as np

def sigmoid(x):
    '''
    激活函数
    :param x:
    :return:
    '''
    return 1 / (1 + np.exp(-x))


def sigmoidDerivation(y):
    return y * (1 - y)


def forward_propagation(x,w,b):
    '''
    前向传播
    :param x: 输入数据
    :param w: w权重
    :param b:
    :return:
    '''
    return np.dot(w,x) + b



if __name__ == '__main__':
    #初始化参数
    alapha = 0.01
    numIter = 100000#迭代次数
    #随机初始化权重
    w1 = np.random.rand(2,2)
    w2 = np.random.rand(2,2)
    print(w1)
    b1 = np.random.random()
    b2 = np.random.random()
    print(b1,b2)
    x = [0.05,0.10]
    y = [0.01,0.98]
    #前向传播
    z1 = forward_propagation(x,w1,b1)
    a1 = sigmoid(z1)
    z2 = forward_propagation(x,w2,b2)
    a2 = sigmoid(z2)
    for n in range(numIter):
        #反向传播 使用损失函数 c = 1 / (2n) * sum[(y - a2)^ 2]
        #分为两次
        #首先对最后一层进行误差计算
        delta2 = np.multiply(-(y - a2),np.multiply(a2,1-a2))
        print(delta2)
        #计算前一层的错误
        delta1 = np.multiply(np.dot(np.array(w2).T,delta2),np.multiply(a1,1-a1))
        print(delta1)
        #更新权重
        for i in range(len(a2)):
            w2[i] = w2[i] - alapha * delta2[i] * a1
        for i in range(len(a1)):
            w1[i] = w1[i] - alapha * delta1[i] * np.array(x)
        #继续前向传播，算出误差值
        z1 = forward_propagation(x,w1,b1)
        a1 = sigmoid(z1)
        z2 = forward_propagation(x, w2, b2)
        a2 = sigmoid(z2)
        print(n,":","result:",a2[0],a1[0])