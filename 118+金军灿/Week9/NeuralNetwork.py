import numpy as np
from scipy import special
import matplotlib.pyplot as plt


# 设计网络模型
class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设计的是一个包含输入层，隐藏层，输出层的三层网络
        self.inputnodes = inputnodes
        self.hiddennodes = hiddennodes
        self.outputnodes = outputnodes

        # 设置模型梯度更新的学习率参数
        self.lr = learningrate

        """
            三层网络需要初始化两个权重矩阵
            wih:表示输入层到隐藏层的权重矩阵
            who:表示隐藏层到输出层的权重矩阵
        """
        self.wih = np.random.randn(self.hiddennodes, self.inputnodes)
        self.who = np.random.randn(self.outputnodes, self.hiddennodes)

        """
            定义模型使用的激活函数
        """
        self.activation_function = lambda x: special.expit(x)

    def train(self, input_array, target_array):
        """
        定义网络模型前向传播的计算过程,进行梯度反向传播，实现一次迭代过程
        Args:
            input_array:训练数据
            target_array:真实的标签数据
        """
        input_array = np.array(input_array, ndmin=2).T
        target_array = np.array(target_array, ndmin=2).T
        hidden_inputs = np.dot(self.wih, input_array)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算回传的梯度信息
        outputs_error = target_array - final_outputs
        hidden_error = np.dot(self.who.T, outputs_error * final_outputs * (1 - final_outputs))

        # 权重参数进行更新
        self.who += self.lr * np.dot(outputs_error * final_outputs * (1 - final_outputs), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot(hidden_error * hidden_outputs * (1 - hidden_outputs), np.transpose(input_array))

    def query(self, input_array):
        """
        定义网络模型前向传播的计算过程
        Args:
            input_array:训练数据
        Returns:
            返回网络计算输出结果
        """
        hidden_inputs = np.dot(self.wih, input_array)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        print(final_outputs)

        return final_outputs


# 初始化网络参数
"""
    输入图像数据为28*28的手写数字图片，将其reshape为一维向量，长度为784
    由于是十分类任务，最后输出的结点数目是10
    中间隐藏层结点数可以根据实验效果进行更改
"""
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.1
net = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# 读取csv文件里面的训练数据
data_file = open("dataset/mnist_train.csv")
data_list = data_file.readlines()
data_file.close()

for record in data_list:
    all_values = record.split(",")
    inputs = (np.asfarray(all_values[1:])) / 225.0 * 0.99 + 0.01
    targets = np.zeros(output_nodes)+0.01
    targets[int(all_values[0])] = 0.99
    net.train(inputs,targets)


'''
在原来网络训练的基础上再加上一层外循环
但是对于普通电脑而言执行的时间会很长。
epochs 的数值越大，网络被训练的就越精准，但如果超过一个阈值，网络就会引发一个过拟合的问题.
'''
#加入epocs,设定网络的训练循环次数
epochs = 50

for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        net.train(inputs, targets)

'''
最后我们把所有测试图片都输入网络，看看它检测的效果如何
'''
test_data_list = open("dataset/mnist_train.csv")
test_data_list = test_data_list.readlines()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字,推理
    outputs = net.query(inputs)
    #找到数值最大的神经元对应的 编号
    label = np.argmax(outputs)
    print("output reslut is : ", label)
    #print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)