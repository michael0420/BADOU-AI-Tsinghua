# -*- coding: utf-8 -*-

import numpy as np
"""
单层神经网络简化版
只支持一个隐藏层，激活函数仅为Sigmoid
偏置B未支持
"""
class NeuralNetwork:
    #初始化网络，包括输入层节点数，隐藏层节点数，输出层节点数，学习率
    def __init__(self,inputnodes,hiddennodes,outputnodes,lrate):
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes
        self.lrate = lrate
        # 初始化权值矩阵
        self.wih = (np.random.normal(0.0, pow(self.hiddennodes,-0.5), (self.hiddennodes,self.inputnodes) )  )
        self.who = (np.random.normal(0.0, pow(self.outputnodes,-0.5), (self.outputnodes,self.hiddennodes) )  )
    #训练神经网络,包括输入矩阵X和输出矩阵Y
    def train(self,X,Y):
        X = np.reshape(X, (len(X),1))
        Y = np.reshape(Y,(len(Y),1))

        #计算隐藏层的输出
        h1 = self.activate(X,self.wih)
        out = self.activate(h1,self.who)

        #逆向求导
        output_error = out-Y
        hidden_error = np.dot(self.who.T,output_error*out*(1-out))
        self.who -= self.lrate * np.dot(output_error*out*(1-out),h1.T)
        self.wih -= self.lrate * np.dot(hidden_error*h1*(1-h1),X.T)

    #正向求值
    def activate(self,X,W):
        #X@W
        re = np.dot(W,X)
        out = self.sigmoid(re)
        return out

    #sigmoid激活函数
    def sigmoid(self,Z):
        out = 1/(1+np.exp(-Z))
        return out

    def  query(self,inputs):
        #根据输入数据计算并输出答案
        #计算中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activate(inputs,self.wih)
        #计算最外层神经元经过激活函数后输出的信号量
        final_outputs = self.activate(hidden_outputs,self.who)
        print(final_outputs)
        return final_outputs
"""
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
"""
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#加入epocs,设定网络的训练循环次数
epochs = 10
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字
    outputs = n.query(inputs)
    #找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)