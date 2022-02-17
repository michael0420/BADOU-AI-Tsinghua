"""
基于numpy的全连接神经网络实现
注：网络有三层，输入层、1个隐藏层、输出层
"""
import numpy as np
import os
import cv2 as cv


class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # 初始化网络，设置输入层，中间层，和输出层节点数
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # 设置学习率
        self.lr = learning_rate
        self.lr = learning_rate

        # 初始化权重矩阵，将权重初始化为-0.5~0.5(覆盖正数、0、复数)之间的浮点数
        # self.weights_i2h = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
        # self.weights_h2o = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
        self.Wih = (np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes)))
        self.Who = (np.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes)))

        # 初始化偏置矩阵，将偏置矩阵初始化为0
        self.bih = np.zeros((self.hidden_nodes, 1), dtype=np.float32)
        self.bho = np.zeros((self.output_nodes, 1), dtype=np.float32)

        # 定义激活函数，采用sigmoid函数：y = 1 / (1 + e^(-x))
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

        # 设置损失函数，采用均方差损失函数
        self.loss_function = lambda Yp, Yt: np.sum(np.square(Yt - Yp)) / Yp.shape[0]

    def train(self, X, Yt):
        # 1.前向传播
        # 输入层——>隐藏层
        Zih = np.dot(self.Wih, X.T) + self.bih   # 加权求和: W.*X+b，隐藏层的输入
        Ah = self.activation_function(Zih)  # 经隐藏层激活后输出
        # 隐藏层——>输出层
        Zho = np.dot(self.Who, Ah) + self.bho  # 加权求和
        Yp = self.activation_function(Zho)  # 经输出层激活后输出整个网络计算结果

        # 2.反向传播，权重更新
        # 总体误差
        loss = self.loss_function(Yp, Yt)
        n = Yp.shape[0]
        # 计算梯度（偏导数），更新权重
        # 输出层——>隐藏层
        grad_Who = np.dot(-2 / n * (Yt - Yp) * Yp * (1 - Yp), np.transpose(Ah))
        self.Who = self.Who - self.lr * grad_Who

        grad_bho = -2 / n * (Yt - Yp) * Yp * (1 - Yp)
        self.bho = self.bho - self.lr * grad_bho

        # 隐藏层——>输入层
        grad_Ah = np.dot(np.transpose(self.Who), -2 / n * (Yt - Yp) * Yp * (1 - Yp))
        grad_Wih = np.dot(grad_Ah * Ah * (1 - Ah), X)
        self.Wih = self.Wih - self.lr * grad_Wih

        grad_bih = grad_Ah * Ah * (1 - Ah)
        self.bih = self.bih - self.lr * grad_bih
        return loss

    def predict(self, X, Yt):
        # 前向传播做推理预测
        Zih = np.dot(self.Wih, X.T)
        Ah = self.activation_function(Zih)
        Zho = np.dot(self.Who, Ah)
        Yp = self.activation_function(Zho)
        loss = self.loss_function(Yp, Yt)
        return Yp, loss

    def per_line_data_preprocess(self, data_line):
        raw_data = data_line.split(',')
        # raw_data包含一个样本的标签（第一个值）和特征（其他值）
        # 获取label，并将label转为one-hot编码
        label = np.zeros((self.output_nodes, 1))
        label[int(raw_data[0])] = 1.0
        # 获取特征，转为数组, 并将数据归一化至(0.01, 1)之间，shape=()
        feature = np.asfarray(raw_data[1:])     # numpy.asfarray()返回转换为浮点类型的数组，asfarray= as float array
        feature = feature / 255.0 * 0.99 + 0.01
        feature = feature.reshape((1, self.input_nodes))
        return feature, label


def array_to_image(test_X, index):
    # 将测试集数据转为图片保存，便于查看
    tmp_X = (test_X - 0.01) * 255.0 / 0.99
    gray_values = np.array(tmp_X, dtype=np.uint8).reshape((28, 28))
    save_folder = '../00-data/datasets/my_mnist/test_images'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_path = os.path.join(save_folder, str(index) + '.png')
    cv.imwrite(save_path, gray_values)


if __name__ == '__main__':
    # 初始化网络
    '''
    由于一张图片总共有28*28=784个数值，因此我们需要让网络的输入层具备784个输入节点
    '''
    input_nodes = 784
    hidden_nodes = 512
    output_nodes = 10
    learning_rate = 0.1
    epochs = 100
    MLP = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    # 读入原始训练集数据
    train_csv_path = "../00-data/datasets/my_mnist/mnist_train.csv"
    with open(train_csv_path, 'r') as f:
        train_data_lines = f.readlines()  # 每一行为一个样本的数据

    # 训练：迭代训练网络
    for epoch in range(epochs):
        train_total_loss = 0
        # 对每个样本数据进行预处理，然后送入网络训练
        for train_data_line in train_data_lines:
            train_X, train_Yt = MLP.per_line_data_preprocess(train_data_line)
            train_step_loss = MLP.train(train_X, train_Yt)
            train_total_loss += train_step_loss
        print('training, epoch: {}, loss: {}'.format(epoch, round(train_total_loss, 5)))

    print("--------------------训练结束-----------------------")

    # 预测，对测试集数据进行推理预测
    test_labels = []
    test_predicts = []
    test_scores = []
    test_total_loss = 0
    idx = 0

    test_csv_path = '../00-data/datasets/my_mnist/mnist_test.csv'
    with open(test_csv_path, 'r') as f:
        test_data_lines = f.readlines()  # 每一行为一个样本的数据

    for test_data_line in test_data_lines:
        # 测试集数据预处理
        test_X, test_Yt = MLP.per_line_data_preprocess(test_data_line)
        y_label = np.argmax(test_Yt)
        test_labels.append(y_label)
        # 附加：将测试集数据转为图片保存
        array_to_image(test_X, idx)
        idx += 1
        # 预测
        test_Yp, test_step_loss = MLP.predict(test_X, test_Yt)
        y_predict = np.argmax(test_Yp)
        test_predicts.append(y_predict)
        test_total_loss += test_step_loss
        # 统计得分
        if y_predict == y_label:
            test_scores.append(1)
        else:
            test_scores.append(0)
    print('test dataset, labels: ', test_labels)
    print('test dataset, predicts: ', test_predicts)
    print('test scores: ', test_scores)

    # 计算图片判断的成功率
    print("test, accuracy: {}, loss: {}".format(sum(test_scores) / len(test_scores), test_total_loss))

