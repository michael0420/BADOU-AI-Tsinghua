#coding:utf-8

'''
训练过程分两步：
第一步是计算输入训练数据，给出网络的计算结果，这点跟我们前面实现的query()功能很像。
第二步是将计算结果与正确结果相比对，获取误差，采用误差反向传播法更新网络里的每条链路权重。
'''
import numpy as np


def train(self, inputs_list, targets_list):
    # 根据输入的训练数据更新节点链路权重
    '''
    :param self:
    :param inputs_list: 输入的训练数据
    :param targets_list: 训练数据对应的正确结果
    :return:
    '''
    inputs = np.array(inputs_list, ndmin=2).T   #ndmin=定义数组的最小维度
    targets = np.array(targets_list, ndim=2).T
    #计算信号经过输入层后产生的信号量
    hidden_inputs = np.dot(self.Wih, inputs)
    hidden_outputs = self.activation(hidden_inputs)
    #输出层接受来自中间层的信号量
    final_inputs = np.dot(self.Who, hidden_outputs)
    final_outputs = self.activation(final_inputs)

'''
[2]
上面代码根据输入数据计算出结果后，我们先要获得计算误差.
误差就是用正确结果减去网络的计算结果。
在代码中对应的就是(targets - final_outputs)
'''
def train(self, inputs_list, targets_list):
    # 根据输入的训练数据更新节点链路权重
    '''
    :param self:
    :param inputs_list: 输入的训练数据
    :param targets_list: 训练数据对应的正确结果
    :return:
    '''
    inputs = np.array(inputs_list, ndmin=2).T   #ndmin=定义数组的最小维度
    targets = np.array(targets_list, ndim=2).T
    #计算信号经过输入层后产生的信号量
    hidden_inputs = np.dot(self.Wih, inputs)
    hidden_outputs = self.activation(hidden_inputs)
    #输出层接受来自中间层的信号量
    final_inputs = np.dot(self.Who, hidden_outputs)
    final_outputs = self.activation(final_inputs)

    #计算误差
    output_errors = targets - final_outputs
    hidden_errors = np.dot(self.Who.T, output_errors * final_outputs * (1-final_outputs))   #隐藏层输出的梯度
    # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
    self.Who += self.lr * np.dot((output_errors * final_outputs * (1-final_outputs)), np.transpose(hidden_outputs))
    self.Wih += self.lr * np.dot((hidden_errors * hidden_outputs *(1-hidden_outputs)), np.transpose(inputs))

    pass

'''使用实际数据来训练我们的神经网络'''
#open函数里的路径根据数据存储的路径来设置
data_file = open('dataset/mnist_test.csv')
data_list = data_file.readlines()   #readlines()读取所有行然后把它们作为一个字符串列表返回
data_file.close()
len(data_list)
print(data_list[0])
#【3】
'''
这里可以利用画图.py将输入绘制出来
'''