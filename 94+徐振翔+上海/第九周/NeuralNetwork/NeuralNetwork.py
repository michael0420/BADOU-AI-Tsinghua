import numpy as np
import scipy.special


# def relu(x):
#     return x * (x > 0)


class NeuralNetwork():
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inode = input_nodes  # 输入层
        self.hnode = hidden_nodes  # 隐藏层
        self.onode = output_nodes  # 输出层
        self.lr = learning_rate  # 学习率
        # 输入层和隐藏层间的权重矩阵，
        # h = w @ i
        # 注意：(hnode,1) = (hnode,inode) * (inode,1)
        # self.wih = np.random.randn(self.hnode, self.inode) - 0.5
        # 隐藏层和输出层间的权重矩阵 (onode,1) = (onode,hnode) * (hnode,1)
        # self.who = np.random.randn(self.onode, self.hnode) - 0.5
        self.wih = (np.random.normal(0.0, pow(self.hnode, -0.5), (self.hnode, self.inode)))
        self.who = (np.random.normal(0.0, pow(self.onode, -0.5), (self.onode, self.hnode)))
        # print("wih\n", self.wih)
        # print("wih\n", self.who)
        # 激活函数
        self.activation_function = lambda x: scipy.special.expit(x)  # sigmoid
        # self.activation_function = lambda x: relu(x)  # relu

    def train(self, input_data, labels):
        # 调整数据结构
        i_o = input_data.reshape(-1, 1)
        l_list = labels.reshape(-1, 1)
        h_o, o_o = self.forward_progress(i_o)  # 前向传播
        o_err = l_list - o_o  # 真实结果与最终输出结果差值列表
        h_err = np.dot(self.who.T, (o_err * o_o * (1 - o_o)))  # -who的梯度： do/dwho = h_0*(-(l-o_o)*o_o*(1-o_o))
        # 根据梯度更新权重
        self.who += self.lr * np.dot((o_err * o_o * (1 - o_o)), np.transpose(h_o))
        self.wih += self.lr * np.dot((h_err * h_o * (1 - h_o)), np.transpose(i_o))
        # print(self.who[:3][0])
        # print(self.wih[:3][0])

    def predict(self, input_data):
        i_d = input_data
        # print('-' * 20)
        # print(i_d.shape)
        h_o, o_o = self.forward_progress(i_d)  # 前向传播
        return o_o  # 输出结果

    def forward_progress(self, input_data):
        # input给到hidden的输入
        h_i = np.dot(self.wih, input_data)
        # hidden激活后的输出
        h_o = self.activation_function(h_i)
        # hidden给到output的输入
        o_i = np.dot(self.who, h_o)
        # output激活后的输出
        o_o = self.activation_function(o_i)
        return h_o, o_o  # 返回隐藏层和输出层结果，隐藏层用于反向传播


if __name__ == '__main__':
    # 初始化网络
    input_node = 784
    hidden_node = 200
    output_node = 10
    learning_rate = 0.1
    model = NeuralNetwork(input_node, hidden_node, output_node, learning_rate)
    # model = NeuralNetwork(2, 3, 1, learning_rate)
    # 读入训练集数据
    with open('dataset/mnist_train.csv') as train_file:
        train_values = train_file.readlines()
        # train_datas = train_file.readlines()
    # print(train_datas[:1])
    # print(type(train_datas))
    # print(train_datas[:10])
    # 设置epochs次数
    epochs = 10
    for i in range(epochs):
        # cnt = 0
        # 逐行解析数据并训练
        for datas in train_values:
        # for datas in train_datas:
            if (type(datas) == str):
                # print(type(datas))
                # cnt += 1
                # print(cnt)
                # if (cnt == 100):
                #     cnt += 1
                # 以‘,’区分数据并读入 区分数据并读入,移除所有换行符，防止csv读入出错
                all_data = datas.strip().split(',')
                # print(len(all_data))
                # 对图像数据进行归一化处理,图像从1开始，第0个是图像本身的label，本身读入的就是一维数组，所以不需要拍扁处理
                train_datas = np.asfarray(all_data[1:]) / 255 * 0.99 + 0.01
                # 设置标签与图像的onehot映射
                train_labels = np.zeros(output_node)
                train_labels[int(all_data[0])] = 1
                # train_labels = np.zeros(output_node)+0.01
                # train_labels[int(all_data[0])] = 0.99
                # train_labels[int(train_datas[0])] = 1 # 之前测试结果一直为0，是因为把几乎所有输入的标签都给成0了。
                # 训练模型
                model.train(train_datas, train_labels)
    print(model.who[0][:10],model.wih[0][:10])
    # 读入测试集数据
    with open('dataset/mnist_test.csv') as test_file:
        test_values = test_file.readlines()
        # test_datas = test_file.readlines()
    scores = []
    # 逐行解析数据并预测
    cnt = 0
    for t_datas in test_values:
    # for datas in test_datas:
        # 以‘,’区分数据并读入,移除所有换行符，防止csv读入出错
        t_all_data = t_datas.strip().split(',')
        # 对图像数据进行归一化处理,图像从1开始，第0个是图像本身的label，本身读入的就是一维数组，所以不需要拍扁处理
        test_datas = np.asfarray(t_all_data[1:]) / 255 * 0.99 + 0.01
        # 获取预测数据结果
        test_labels = model.predict(test_datas)
        # 最大数字对应的索引为
        test_label = np.argmax(test_labels)
        # 正确标签
        correct_label = int(t_all_data[0])
        print("该图片实际对应的数字是:", correct_label)
        print("该图片预测的结果是:", test_label)
        if correct_label == test_label:
            scores.append(1)
        else:
            scores.append(0)
        scores_np = np.array(scores)
        print(test_labels)
        print("scores\n", scores)
        print("perfermance(准确率) =", scores_np.sum() / scores_np.size)
