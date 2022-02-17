import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes  # 输入结点个数
        self.hnodes = hiddennodes  # 隐藏层结点个数
        self.onodes = outputnodes  # 输出结点个数
        self.lr = learningrate  # 学习率
        # 初始化权重
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes)))
        self.who = (numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes)))
        self.activation_function = lambda x: scipy.special.expit(x)  # 激活函数
        pass

    def train(self, input_list, targets_list):
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors % final_outputs * (1 - final_outputs))
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    def query(self, inputs):
        hidden_inputs = numpy.dot(self.wih, inputs)  # 计算隐藏层输入量
        hidden_outputs = self.activation_function(hidden_inputs)  # 计算激活量
        final_inputs = numpy.dot(self.who, hidden_outputs)  # 最终层输入
        final_outputs = self.activation_function(final_inputs)
        print(final_outputs)
        return final_outputs


def NNinit(epoches):
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.1
    n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    training_data_file = open("dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    test_data_file = open("dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    training_data_file.close()  # 关闭文件占用内存
    test_data_file.close()
    for i in range(epoches):
        for record in training_data_list:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:])) / 255.0  # 归一化数据
            targets = numpy.zeros(output_nodes)
            targets[int(all_values[0])] = 1
            n.train(inputs, targets)  # 训练
    scores = []
    for record in test_data_list:
        all_values = record.split(',')  # 存储数据
        correct_number = int(all_values[0])
        print("该图片对应的数字为:", correct_number)
        inputs = (numpy.asfarray(all_values[1:])) / 255.0
        outputs = n.query(inputs)  # 查询
        label = numpy.argmax(outputs)
        print(outputs)
        print("该网络认为该数字为:", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)
    scores_array = numpy.asarray(scores)
    print("performence = ", scores_array.sum() / scores_array.size)


NNinit(5)
