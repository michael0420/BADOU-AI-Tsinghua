#!/usr/bin/env python 
# coding:utf-8
import numpy as np
import scipy.special


class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 输入层
        self.input_nodes = input_nodes
        # 隐藏层
        self.hidden_nodes = hidden_nodes
        # 输出层
        self.output_nodes = output_nodes
        # 学习率
        self.learning_rate = learning_rate
        # 输入层和隐藏层之间的权重矩阵
        self.wih = np.random.randn(self.hidden_nodes, self.input_nodes) / np.sqrt(self.hidden_nodes)
        # 隐藏层和输出层之间的权重矩阵
        self.who = np.random.randn(self.output_nodes, self.hidden_nodes) / np.sqrt(self.output_nodes)
        # 激活函数,sigmoid
        self.activation_func = lambda x: scipy.special.expit(x)

    # 训练模型
    def train(self, train_data, train_labels):
        # 转为二维结构
        input_array = np.array(train_data, ndmin=2).T
        train_label_list = np.array(train_labels, ndmin=2).T
        # 隐藏层和输出层列表
        hidden_output_list, final_output_list = self.fp_func(input_array)
        # 预测值和真实值的差值
        output_error_list = train_label_list - final_output_list
        # 隐藏层的误差
        hidden_error_list = np.dot(self.who.T, (output_error_list * final_output_list * (1 - final_output_list)))
        # 更新网络中的权重
        self.who += self.learning_rate * np.dot((output_error_list * final_output_list * (1 - final_output_list)),
                                                hidden_output_list.T)
        self.wih += self.learning_rate * np.dot((hidden_error_list * hidden_output_list * (1 - hidden_output_list)),
                                                input_array.T)

    # 预测
    def predict(self, input_data):
        input_array = np.array(input_data, ndmin=2).T
        hidden_output_list, final_output_list = self.fp_func(input_array)
        return hidden_output_list, final_output_list

    # 正向传播
    def fp_func(self, input_data):
        # 输入层到隐藏层,加权求和
        hidden_input_list = np.dot(self.wih, input_data)
        # 隐藏层激活函数
        hidden_output_list = self.activation_func(hidden_input_list)
        # 隐藏层到输出层,加权求和
        output_input_list = np.dot(self.who, hidden_output_list)
        # 输出层激活函数
        final_output_list = self.activation_func(output_input_list)
        return hidden_output_list, final_output_list


if __name__ == '__main__':
    # 初始化参数
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    network = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # 加载数据
    with open("/Users/lmc/Documents/workFile/AI/week9/job/dataset/mnist_train.csv", 'r') as train_data:
        train_data_file = train_data.readlines()
    # 设置训练次数
    epoch = 5
    for each in range(epoch):
        for record in train_data_file:
            split_data = record.split(",")
            # 数据归一化处理 [0.01,1]
            train_data_list = np.asfarray(split_data[1:]) / 255 * 0.99 + 0.01
            # 将真实标签映射为独热编码
            train_labels = np.zeros(output_nodes)
            train_labels[int(split_data[0])] = 1
            # 训练模型
            network.train(train_data_list, train_labels)
    # 通过测试集查看模型效果
    with open("/Users/lmc/Documents/workFile/AI/week9/job/dataset/mnist_test.csv", 'r') as train_data:
        train_data_file = train_data.readlines()
    score_list = []
    for record in train_data_file:
        split_data = record.split(",")
        # 真实标签
        real_label = int(split_data[0])
        print("该图片实际对应的数字是:", real_label)
        # 数据归一化处理 [0.01,1]
        test_data_list = np.asfarray(split_data[1:]) / 255 * 0.99 + 0.01
        hidden_output_list, final_output_list = network.predict(test_data_list)
        # 最大数字对应的索引
        max_idx = np.argmax(final_output_list)
        print("神经网络认为该图片对应的数字是:", max_idx)
        # 如果预测值和真实值一致,则追加1,否则0
        if real_label == max_idx:
            score_list.append(1)
        else:
            score_list.append(0)
    print("score_list", score_list)
    print("该神经网络的精度:", sum(score_list) / len(score_list))
