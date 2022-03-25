#coding:utf-8

'''模型网络搭建，手写数字识别
自己动手，训练+推理
'''

import numpy as np
import scipy.special


#网络模型
class NeuralNetWork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learningrate):
        # 初始化网络
        self.in_node = input_nodes
        self.hid_node = hidden_nodes
        self.out_node = output_nodes
        self.lr = learningrate
        # 初始化。准确率不稳定
        # self.Wih = np.random.rand(self.hid_node, self.in_node) - 0.5
        # self.Who = np.random.rand(self.out_node, self.hid_node) - 0.5
        # 初始化
        self.Wih = (np.random.normal(0.0, pow(self.hid_node, -0.5), (self.hid_node, self.in_node)))  # 均值、标准差
        self.Who = (np.random.normal(0.0, pow(self.out_node, -0.5), (self.out_node, self.hid_node)))

        # sigmod激活函数
        self.acti_function = lambda x:scipy.special.expit(x)    #def activation_function(self, x):/n return scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T   #生成二维矩阵
        targets = np.array(targets_list, ndmin=2).T
        # 前向
        hidden_inputs = np.dot(self.Wih, inputs)
        hidden_outputs = self.acti_function(hidden_inputs)
        final_inputs = np.dot(self.Who, hidden_outputs)
        final_outputs = self.acti_function(final_inputs)
        # loss
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.Who.T, output_errors * final_outputs * (1-final_outputs))
        # 更新梯度
        self.Who += self.lr * np.dot((output_errors * final_outputs * (1-final_outputs)), np.transpose(hidden_outputs))
        self.Wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1-hidden_outputs)), np.transpose(inputs))

    def query(self, inputs):
        # 前馈网络
        hidden_inputs = np.dot(self.Wih, inputs)
        hidden_outputs = self.acti_function(hidden_inputs)
        final_inputs = np.dot(self.Who, hidden_outputs)
        final_outputs = self.acti_function(final_inputs)
        print('final_outputs:', final_outputs)
        return final_outputs

#主函数
def main():
    # 初始化网络
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learningrate = 0.1
    n = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learningrate)
    # 读入训练数据
    training_file = open('dataset/mnist_train.csv', 'r')
    training_list = training_file.readlines()   #['1,0,0,../n', '2,0,0,../n', '',...]。 scripts
    training_file.close()
    # epoch， 设置训练循环次数
    epochs = 5
    for e in range(epochs):
        for record in training_list:    #['1,0,0,../n', '', ...]
            all_values = record.split(',')  #所有的数值
            inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01  ##asfarray()把普通数组转换成浮点型数组,归一化到0.01-1之间
            # 设置图片与数值的对应关系
            targets = np.zeros(output_nodes) + 0.01
            targets[int(all_values[0])] = 0.99
            # 训练
            n.train(inputs, targets)

    # test
    test_files = open('dataset/mnist_test.csv')
    test_list = test_files.readlines()
    test_files.close()
    scores = []
    for record in test_list:
        all_values = record.split(',')
        correct_numbers = int(all_values[0])
        print('该图片对应的数字为：', correct_numbers)
        # 图片预处理
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 推理
        outputs = n.query(inputs)
        label = np.argmax(outputs)      #返回一个numpy数组中最大值的索引值
        print('网络认为的数值为：', label)
        if label == correct_numbers:
            scores.append(1)
        else:
            scores.append(0)
    print('scores: ', scores)
    # 计算精确度
    scores_array = np.asfarray(scores)
    perfect = scores_array.sum() / scores_array.size    # 总和数/总数
    print('perfermance:', perfect)

if __name__ == '__main__':
    main()