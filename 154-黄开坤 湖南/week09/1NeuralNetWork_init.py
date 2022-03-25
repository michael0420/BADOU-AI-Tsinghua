#coding:utf-8

import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #初始化网络，设置输入层，中间层，和输出层节点个数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 设置学习率
        self.lr = learningrate
        #初始化权重
        self.Wih = np.random.rand(self.hnodes, self.inodes) - 0.5   #输入到中间
        self.Who = np.random.rand(self.onodes, self.hnodes) - 0.5   #中间到输出
        # 激活函数
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    def train(self):
        # 根据输入的训练数据更新节点链路权重
        pass

    def query(self, inputs):
        # 根据输入数据计算并输出答案
        # 计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.Wih, inputs)
        # 中间层经过激活函数后形成的输出信号量
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接受的信号量
        final_inputs = np.dot(self.Who, hidden_outputs)
        # 过激活函数
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs

#for example
input_nodes = 3
hidden_nodes = 3
output_nodes = 3

learning_rate = 0.3
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
n.query([1.0, 0.5, -1.5])