import numpy as np
import scipy.special


class NeuralNetWork:
    """一个全链接神经网络"""

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        """
        初始化网络，设置输入层，中间层，和输出层节点数
        :param input_nodes: 输入层节点数
        :param hidden_nodes: 中间层(隐藏层)节点数
        :param output_nodes: 输出层节点数
        :param learning_rate: 学习率
        """
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        """
        构造并初始化层间权重矩阵。
        根据矩阵乘法。构造的权重矩阵的行数由后层节点数决定，列数由前层节点数决定。
        由于权重不一定都是正的，它完全可以是负数，那么我们在初始化时，不妨把所有权重初始化为-0.5到0.5之间。
        """
        self.wih = np.random.rand(self.hnodes, self.inodes) - 0.5
        # wih矩阵是一个(隐藏层节点数, 输入层节点数)，各元素取值[-0.5, 0.5]的矩阵，符合要求。下同。
        self.who = np.random.rand(self.onodes, self.hnodes) - 0.5
        '''
        scipy.special.expit()对应的是sigmoid函数.
        使用Python保留关键字lambda构造匿名函数lambda x: scipy.special.expit(x)可以直接得到激活函数计算后的返回值。
        '''
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        """
        完成神经网络的训练算法部分
        :param inputs_list: 输入的训练数据
        :param targets_list: 训练数据对应的正确结果
        """
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        # (↓正向传播过程)
        # 数据由输入层向中间层(隐藏层)进行层间传递，按照加权求和的规则计算
        hidden_inputs = np.dot(self.wih, inputs)
        # 数据在中间层(隐藏层)的接收端向输出端进行层内传递，经过激活函数后形成的输出数据矩阵
        hidden_outputs = self.activation_function(hidden_inputs)
        # 数据由中间层(隐藏层)向输出层进行层间传递，按照加权求和的规则计算
        final_inputs = np.dot(self.who, hidden_outputs)
        # 数据在输出层的接收端向输出端进行层内传递，经过激活函数后形成最终的输出数据矩阵
        final_outputs = self.activation_function(final_inputs)
        # (↓反向传播过程)
        """
        这里注意，如下反向传播计算式的形式是由我们使用的损失函数为MSE函数，以及上文提到激活函数为sigmoid函数共同决定的，课程里省略了。
        由于我们设计的神经网络是采用“训练次数”截止模式，因此可以省略判断MSE截止模式相关代码的编写。
        """
        # 计算正向传播输出结果与标签的误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        # 按照链式求导法则求出损失函数MSE对各个权重w的偏导数，依据梯度下降法更新各权重
        """
        想要看懂这个权重更新代码的提示:
        1. 要在数学上实现反向传播过程的推导，得到损失函数MSE对各权重wi的偏导数表达式；
        2. 根据”输入→中间层“和”中间层→输出层“，将偏导数分为两组∂MSE/∂[w1,w2,w3,w4]和∂MSE/∂[w5,w6,w7,w8]；
        3. 将偏导数的多项式表达式形式，转换成矩阵乘法表达式形式；
        4. 转换时尽量做到矩阵形式中每一项的样子与数据在变量中存储形式一致，这样更容易理解和编写代码。
        5. 将某些步骤中“对角阵与列向量乘法”变成了更加容易用代码实现的“数组乘法”
        """
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), np.transpose(inputs))
        # self.wih更新算式中，np.dot()的第一个参数表达式应用了“数组乘法”
        pass

    def query(self, inputs):
        """
        完成神经网络的预测算法部分
        :param inputs: 输入层的输入数据矩阵
        :return: 神经网络一次正向传递的最终输出
        """
        # 数据由输入层向中间层(隐藏层)进行层间传递，按照加权求和的规则计算
        hidden_inputs = np.dot(self.wih, inputs)
        # 数据在中间层(隐藏层)的接收端向输出端进行层内传递，经过激活函数后形成的输出数据矩阵
        hidden_outputs = self.activation_function(hidden_inputs)
        # 数据由中间层(隐藏层)向输出层进行层间传递，按照加权求和的规则计算
        final_inputs = np.dot(self.who, hidden_outputs)
        # 数据在输出层的接收端向输出端进行层内传递，经过激活函数后形成最终的输出数据矩阵
        final_outputs = self.activation_function(final_inputs)
        print(f"神经网络判断输出结果：{final_outputs}")
        return final_outputs

    def network_train(self, data, epoches=5):
        """
        完成整个训练网络的训练过程(权重更新过程)部分
        :param data: 训练集，包含数据和标签
        :param epoches: 数据被遍历次数
        """

        for e in range(epoches):
            for record_ in data:
                all_values_ = record_.split(',')  # 把数据依靠','分割，并分别读入
                """
                接下来可以将数据“归一化”，也就是把所有数值全部转换到0.01到1.0之间。
                由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，就能让数据全部落入到0和1之间。
                有些数值很小，除以255后会变为0，这样“有可能”导致链路权重更新出意想不到的问题。
                所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。
                """
                inputs_ = (np.asfarray(all_values_[1:])) / 255.0 * 0.99 + 0.01  # 首个元素是标签，在inputs读取时要去掉。进行“数据分割”
                # 设置图片与数值的对应关系，ont-hot编码
                targets_ = np.zeros(self.onodes) + 0.01  # 创建一个10个元素的数组，各元素均为0.01
                targets_[int(all_values_[0])] = 0.99  # 在数组中，将等同于数字值的索引的元素替换为0.99。假设数字7，就把索引7(第8个)数字更换为0.99
                self.train(inputs_, targets_)  # 启用训练过程


# 初始化神经网络
inputnodes = 784  # 28*28=784，是一个图片数据的像素个数,因此输入层需要784个节点
hiddennodes = 100  # 100：经验值
outputnodes = 10  # 一共10个数字，用10个节点即可输出one-hot编码对应格式的结果供判断
learningrate = 0.3
n = NeuralNetWork(inputnodes, hiddennodes, outputnodes, learningrate)  # 实例化

# 加载训练数据
training_data_file = open("dataset/mnist_test.csv", 'r')  # 只读模式加载数据，注意检查文件存储路径
training_data_list = training_data_file.readlines()  # 将每一行数据作为一个元素，存储在一个list中
training_data_file.close()  # 关闭文件

# 训练实例化的神经网络
n.network_train(training_data_list, epoches=5)

# 加载测试数据
test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []  # 设定一个列表记录每次判断的得分情况，判断正确存入1，错误存入0
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])  # 提取标签值
    print("该图片对应的数字为:", correct_number)

    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01  # 归一化
    outputs = n.query(inputs)  # 让训练好的神经网络判断图片对应的数字并输出结果
    label = np.argmax(outputs)  # 应用numpy.argmax()函数找到数值最大的神经元对应的编号
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(f"得分记录：\n{scores}")

# 计算图片判断的成功率
scores_array = np.asarray(scores)
print(f"perfermance = {scores_array.sum() / scores_array.size * 100}%")
