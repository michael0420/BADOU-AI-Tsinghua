import numpy as np
import scipy.special as sp
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class nnetwork:
    def __init__(self, innodes, hiddennodes, outnodes, lr):
        self.innodes = innodes
        self.hiddennodes = hiddennodes
        self.outnodes = outnodes
        self.lr = lr

        self.wi = np.random.normal(0.0, pow(self.innodes, -0.5), (self.hiddennodes, self.innodes))
        self.wo = np.random.normal(0.0, pow(self.outnodes, -0.5), (self.outnodes, self.hiddennodes))

        self.activatefun = lambda x:sp.expit(x)

    def train(self, inputlist, outputlist):
        inputs = np.array(inputlist, ndmin=2).T #生成二维数组
        outputs = np.array(outputlist, ndmin=2).T
        hiddeninputs = np.dot(self.wi, inputs)
        hiddenoutputs = self.activatefun(hiddeninputs)
        resultinputs = np.dot(self.wo, hiddenoutputs)
        resultoutputs = self.activatefun(resultinputs)

        totalerror = outputs - resultoutputs
        hiddenerror = np.dot(self.wo.T, totalerror*resultinputs*(1-resultinputs))
        self.wo += self.lr * np.dot((totalerror*resultoutputs*(1-resultoutputs)), np.transpose(hiddenoutputs))
        self.wi += self.lr * np.dot((hiddenerror*hiddenoutputs*(1-hiddenoutputs)), np.transpose(inputs))

    def forecast(self, inputs):
        hiddeninputs = np.dot(self.wi, inputs)
        hiddenoutputs = self.activatefun(hiddeninputs)
        resultinputs = np.dot(self.wo, hiddenoutputs)
        resultoutputs = self.activatefun(resultinputs)

        return resultoutputs

# class tfNetwork:
#     def __init__(self):



if __name__ == '__main__':
    # tfNetwork()
    input = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # 随机数噪声模拟数据
    noise = np.random.normal(0, 0.02, input.shape)
    inputTo = np.square(input) + noise

    x = tf.compat.v1.placeholder(tf.float32, [None, 1])  # 先占位
    y = tf.compat.v1.placeholder(tf.float32, [None, 1])

    w_input = tf.Variable(tf.random.normal([1, 10]))  # 计算输入层wx+b
    b_input = tf.Variable(tf.zeros([1, 10]))
    wb_input = tf.matmul(x, w_input) + b_input
    wb_inputs = tf.nn.tanh(wb_input)

    w_output = tf.Variable(tf.random.normal([10, 1]))  # 计算输出层wx+b
    b_output = tf.Variable(tf.zeros([1, 1]))
    wb_output = tf.matmul(wb_inputs, w_output) + b_output
    wb_outputs = tf.nn.tanh(wb_output)

    loss = tf.reduce_mean(tf.square(y - wb_outputs))  # 计算预测值与真实值误差的平方差
    backwardTrain = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(loss)  # 学习率设置0.001

    # sess = tf.compat.v1.Session()
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        for i in range(5000):
            sess.run(backwardTrain, feed_dict={x: input, y: inputTo})

        predictValue = sess.run(wb_outputs, feed_dict={x: input})

        plt.figure()
        plt.scatter(input, inputTo)  # 真实值的散点图
        plt.plot(input, predictValue, 'r', lw=5)  # 预测值用红色线显示，线宽5
        plt.show()
        # sess.close()

# innodes = 784
# hiddennodes = 200   #经验值
# outnodes = 10
# lr = 0.01
# m_model = nnetwork(innodes, hiddennodes, outnodes, lr)
#
# trainDataFile = open("mnist_train.csv", 'r')
# trainDataList = trainDataFile.readlines()
# trainDataFile.close()
#
# epochs = 10
# for e in range(epochs):
#     for trainli in trainDataList:
#         trainValues = trainli.split(',')
#         trainInputs = (np.asfarray(trainValues[1:]))/255.0*0.99+0.01
#
#         trainTargets = np.zeros(outnodes)+0.01
#         trainTargets[int(trainValues[0])] = 0.99
#         m_model.train(trainInputs, trainTargets)
#
#
# testDataFile = open("mnist_test.csv", 'r')
# testDataList = testDataFile.readlines()
# testDataFile.close()
#
# scores = []
# for testli in testDataList:
#     testValues = testli.split(',')
#     correctValue = int(testValues[0])
#     print("该图片对应的数字是：", correctValue)
#
#     testInputs = (np.asfarray(testValues[1:]))/255.0*0.99+0.01
#     testResults = m_model.forecast(testInputs)
#     label = np.argmax(testResults)
#     print("模型计算的结果是：", label)
#
#     if(label == correctValue):
#         scores.append(1)
#     else:
#         scores.append(0)
# print(scores)
#
# scoresArray = np.asarray(scores)
# print("计算结果正确率百分比：", scoresArray.sum()/scoresArray.size)







