import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,lr):
        # 初始化神经网络，该网络由3层构成，输入层，隐藏层，和输出层
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = lr

        # 初始化各层权重 输入层与隐藏层：w12 ，隐藏层与输出层w23
        # numpy.random.normal(loc=0.0, scale=1.0, size=None) 分布的中值,标准差，输出的形式
        # 相乘方式为(w12 * x) 输出结果为一个列向量（num,1）
        self.w12 = np.random.normal(0.0,pow(self.hnodes,-1),(self.hnodes,self.inodes))
        self.w23 = np.random.normal(0.0,pow(self.onodes,-1),(self.onodes,self.hnodes))
        self.b12 = np.random.normal(0.0,1,(self.hnodes,1))
        self.b23 = np.random.normal(0.0,1,(self.onodes,1))
        # 定义激活函数为sigmoid()
        self.activation_function = lambda x:scipy.special.expit(x)


    def train(self,input_list,target_list):
        # 根据target标签进行数据训练 （转换成二维列矩阵）
        inputs = np.array(input_list,ndmin=2).T
        targets = np.array(target_list,ndmin=2).T

        # 正向推理过程
        # 输入层->隐藏层
        # y = w*x + b
        hidden_inputs = np.dot(self.w12,inputs) + self.b12
        # y = sigmoid(x)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 隐藏层->输出层
        final_inputs = np.dot(self.w23,hidden_outputs) + self.b23
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        # Etotal/final_inputs 求导：1/2*(target-output)^2 = output - target
        output_error = final_inputs - targets
        hidden_error = np.dot(self.w23.T,output_error*final_outputs*(1-final_outputs))
        # 计算各权重
        self.w23 = self.w23 - self.lr * np.dot(output_error
                                               * final_outputs * (1 - final_outputs)
                                               ,np.transpose(hidden_outputs))
        self.w12 = self.w12 - self.lr * np.dot(hidden_error
                                               * hidden_outputs * (1 - hidden_outputs)
                                               ,np.transpose(inputs))
        self.b23 = self.b23 - self.lr * output_error * final_outputs * (1 - final_outputs)
        self.b12 = self.b12 - self.lr * hidden_error * hidden_outputs * (1 - hidden_outputs)
        loss = np.sum(output_error)
        print('loss:',loss)

    def query(self,inputs):
        inputs = np.array(inputs, ndmin=2).T

        # 正向推理过程
        # 输入层->隐藏层
        # y = w*x + b
        hidden_inputs = np.dot(self.w12,inputs) + self.b12
        # y = sigmoid(x)
        hidden_outputs = self.activation_function(hidden_inputs)
        #隐藏层->输出层
        final_inputs = np.dot(self.w23,hidden_outputs) + self.b23
        final_outputs = self.activation_function(final_inputs)


        return final_outputs

# 网络初始化

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
lr = 0.1

net = NeuralNetWork(input_nodes,hidden_nodes,output_nodes,lr)
train_data_file = open('dataset/mnist_train.csv','r')
train_data_list = train_data_file.readlines()
train_data_file.close()

# 训练循环 run

epochs = 5
for e in range(epochs):
    # 数据预处理
    for record in train_data_list:
        # 把数据按‘,’区分，并分别读入
        all_values = record.split(',')    # 字符串分割 第一位为字符标签，后面为图像数据
        # 数据标准化
        # np.asfarray() 将字符数据转化为数字
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        # 设置图片与数值的对应关系
        targets = np.zeros(output_nodes) + 0.01  # 保证每个结果中都有少部分概率被选到
        targets[int(all_values[0])] = 1 - (output_nodes-1) *0.01     # 字符数据转换为数字,在结果最大出填入最大概率)

     # 训练数据
        net.train(inputs,targets)


# 预测结果
test_data_file = open('dataset/mnist_test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()

scores = []

for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print('图片对应的结果为：',correct_number)
    # 预处理图片
    inputs = np.asfarray(all_values[1:]) /255.0 * 0.99 + 0.01
    # 结果预测
    outputs = net.query(inputs)
    # 找到对应的最大概率编号位置
    label = np.argmax(outputs)
    print('模型预测输出的图片结果为：',label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)

print(scores)
# 计算模型准确率
scores_array = np.asarray(scores)
print('模型准确率为：',scores_array.sum()/scores_array.size)

