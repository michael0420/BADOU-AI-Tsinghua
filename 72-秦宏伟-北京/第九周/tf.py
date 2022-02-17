# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

training_data_file = open("mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()


#定义两个placeholder存放输入数据
x=tf.placeholder(tf.float32,[1,784])
y=tf.placeholder(tf.float32,[1,10])

#定义神经网络中间层H1
Weights_L1=tf.Variable(tf.random_normal([784,200]))
biases_L1=tf.Variable(tf.zeros([1,200]))    #加入偏置项
Wx_plus_b_L1=tf.matmul(x,Weights_L1)
L1=tf.nn.tanh(Wx_plus_b_L1)   #加入激活函数

#定义神经网络输出层
Weights_L2=tf.Variable(tf.random_normal([200,10]))
biases_L2=tf.Variable(tf.zeros([1,10]))  #加入偏置项
Wx_plus_b_L2=tf.matmul(L1,Weights_L2)
# a_out=tf.nn.tanh(Wx_plus_b_L2)   #加入激活函数
prediction = tf.nn.softmax(Wx_plus_b_L2)

#定义损失函数（均方差函数）
loss=tf.reduce_mean(tf.square(y-prediction))
#定义反向传播算法（使用梯度下降算法训练）
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())

    epochs = 5
    for e in range(epochs):
        # 把数据依靠','区分，并分别读入
        for record in training_data_list:
            all_values = record.split(',')
            x_data = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
            x_data = np.reshape(x_data, (1,len(x_data)))
            # 设置图片与数值的对应关系
            y_data = np.zeros(10) + 0.01
            y_data[int(all_values[0])] = 0.99
            y_data = np.reshape(y_data, (1,len(y_data)))
            sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # # 获得预测值
    test_data_file = open("mnist_test.csv")
    test_data_list = test_data_file.readlines()
    test_data_file.close()
    scores = []
    for record in test_data_list:
        all_values = record.split(',')
        correct_number = int(all_values[0])
        print("该图片对应的数字为:", correct_number)
        # 预处理数字图片
        inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        inputs = np.reshape(inputs, (1,len(inputs)))
        # 让网络判断图片对应的数字
        outputs = sess.run(prediction, feed_dict={x: inputs})
        # 找到数值最大的神经元对应的编号
        label = np.argmax(outputs)
        print("网络认为图片的数字是：", label)
        if label == correct_number:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)