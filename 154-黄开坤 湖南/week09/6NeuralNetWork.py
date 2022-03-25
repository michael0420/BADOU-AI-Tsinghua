#coding:utf-8

'''简单神经网络模型搭建'''

import numpy as np
import scipy.special
import matplotlib.pyplot as plt

class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        #初始化网络，设置输入层，中间层和输出层节点个数
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        #设置学习率
        self.lr = learningrate
        '''初始化权重矩阵，两个'''
        # # 1、 准确率不稳定
        self.Wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        self.Who = np.random.rand(self.onodes, self.hnodes) - 0.5
        # 2、从正态（高斯）分布中抽取随机样本.
        # self.Wih = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))  #均值、标准差
        # self.Who = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        #每个节点过激活函数，sigmoid激活函数
        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    #训练
    def train(self, inputs_list, targets_list):
        #根据输入的训练数据更新节点链路权重
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        #计算信号经过输入层后产生的信号量
        hidden_inputs = np.dot(self.Wih, inputs)
        #中间层神经元过激活
        hidden_outputs = self.activation_function(hidden_inputs)
        #输出层接受来自中间层的信号量
        final_inputs = np.dot(self.Who, hidden_outputs)
        #输出层过激活
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.Who.T, output_errors * final_outputs * (1-final_outputs)) #f'(x)=a(1-a)   #中间层梯度
        # 根据误差计算链路权重的更新量，然后把更新加到原来的原来的链路权重上
        self.Who += self.lr * np.dot((output_errors * final_outputs * (1-final_outputs)), np.transpose(hidden_outputs))
        self.Wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)), np.transpose(inputs))

        pass

    def query(self, inputs):
        # 根据输入数据计算输出数据
        #计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.Wih, inputs)
        # 中间层输出
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算最外层接收到的信号量
        final_inputs = np.dot(self.Who, hidden_outputs)
        #过激活
        final_outputs = self.activation_function(final_inputs)
        print('final_outputs:', final_outputs)
        return final_outputs

def main():
    #初始化网络
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learningrate = 0.1
    n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learningrate)

    # 读入训练数据
    training_data_file = open('dataset/mnist_train.csv', 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    #加入epoch，设定网络的训练循环次数
    epochs = 5
    for e in range(epochs):
        #把数据依靠‘,’区分，并分别读入
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
            #设置图片与数值的对应关系
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

    #测试
    test_data_file = open('dataset/mnist_test.csv')
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print('该图片对应的数字为：', correct_number)
        #预处理数字图片
        inputs = (np.asfarray(all_values[1:])) / 255.0 *0.99 + 0.01
        #让网络判断图片对应的数字
        outputs = n.query(inputs)
        #找到数值最大的神经元对应的编号
        label= np.argmax(outputs)
        print('网络认为图片的数字是：', label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print('scores:', scores)

    #计算准确率
    scores_array = np.asfarray(scores)
    print('perfermance = ', scores_array.sum() / scores_array.size)

if __name__ == '__main__':
    main()