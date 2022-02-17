#!/usr/bin/env python 
# coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 定义x
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
# y的值
y_data = np.square(x_data) + noise
# 占位符定义变量
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# 隐藏层
weight_L1 = tf.Variable(tf.random_normal([1, 10]))
bias_L1 = tf.Variable(0.2)
net_L1 = tf.matmul(x, weight_L1) + bias_L1
# 激活函数
output_L1 = tf.nn.tanh(net_L1)

# 输出层
weight_L2 = tf.Variable(tf.random_normal([10, 1]))
bias_L2 = tf.Variable(0.3)
net_L2 = tf.matmul(output_L1, weight_L2) + bias_L2
output_L2 = tf.nn.tanh(net_L2)

# 损失函数
loss = tf.reduce_mean(tf.square(y - output_L2))
# 更新模型参数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 创建会话
with tf.Session() as sess:
    # 初始化参数
    sess.run(tf.global_variables_initializer())
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 预测
    predict = sess.run(output_L2, feed_dict={x: x_data})
    # 绘图
    plt.figure()
    # 真实值绘制散点图
    plt.scatter(x_data, y_data)
    # 预测值绘制曲线图
    plt.plot(x_data, predict, "r-", lw=1)
    plt.show()
