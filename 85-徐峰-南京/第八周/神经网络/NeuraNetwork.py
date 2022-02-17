import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennode, outputnodes, learingrate):

        #输入
        self.inodes = inputnodes
        self.hnodes = hiddennode
        self.onodes = outputnodes

        #学习率
        self.lr = learingrate

        """
        初始化权重矩阵，两个权重矩阵：
        weight_input_hidden
        weight_hidden_output
        """
        #为什么要这么初始化权重
        self.weight_input_hidden = (np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.weight_hidden_output = (np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))

        #激活函数
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self, input_list, target_list):
        # 根据输入的训练数据更新节点链路权重
        '''
        把inputs_list, targets_list转换成numpy支持的二维矩阵
        .T表示做矩阵的转置
        '''
        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        #信号经过输入层后产生的信号量,
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        #中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_output = self.activation_function(hidden_inputs)

        #输出层
        output = np.dot(self.weight_hidden_output, hidden_output)
        #激活函数
        output = self.activation_function(output)

        #计算误差
        output_errors = targets - output
        hidden_errors = np.dot(self.weight_hidden_output.T, output_errors * output*(1 - output))

        #根据误差计算权重的更新量，然后把更新加到原来的权重上
        self.weight_hidden_output += self.lr * np.dot((output_errors * output * (1 - output)),
                                                     np.transpose(hidden_output))
        self.weight_input_hidden += self.lr * np.dot((hidden_errors * hidden_output * (1 - hidden_output)),
                                                    np.transpose(inputs))
        pass

    def qurey(self, inputs):
        #根据输入数据计算输出答案
        #计算中间层从输入层接收到的信号量
        hidden_inputs = np.dot(self.weight_input_hidden, inputs)
        #激活函数
        hidden_outputs = self.activation_function(hidden_inputs)

        #计算最外层接收到的信号量
        final_inputs = np.dot(self.weight_hidden_output, hidden_outputs)
        #最终输出
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


#初始化网络
'''
由于一张图片总共有28*28 = 784个数值，因此我们需要让网络的输入层具备784个输入节点
'''
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#加入epocs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
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
    outputs = n.qurey(inputs)
    #找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = np.array(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
print(np.random.randn(1, 10))
