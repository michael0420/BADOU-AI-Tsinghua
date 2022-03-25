import tensorflow.compat.v1 as tf
import numpy as np
from matplotlib import pyplot as plt

'''
准备input output
'''
# 200个随机数
input = np.linspace(-0.5, 0.5, 200)
# 对齐维度, 与 np.reshare(200,1) 效果一致
input = input[:, np.newaxis]
noise = np.random.normal(0, 0.02, input.shape)
output = np.square(input) + noise

'''
设计model
'''
# input output 对应的占位符  后面feed_dict要用
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# input->layer1
weight1 = tf.Variable(tf.random_normal([1, 10]))
bias1 = tf.Variable(tf.zeros([1, 10]))
result1 = tf.matmul(x, weight1) + bias1
result1 = tf.nn.tanh(result1)

# layer1->output
weight2 = tf.Variable(tf.random_normal([10, 1]))
bias2 = tf.Variable(tf.zeros([1, 1]))
prediction = tf.matmul(result1, weight2) + bias2
prediction = tf.nn.tanh(prediction)

# loss 使用均方差
loss = tf.reduce_mean(tf.square(prediction - y))
# opt 使用GD
opt = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as session:
    # 网络中的参数初始化
    session.run(tf.global_variables_initializer())

    # train
    for i in range(2000):
        session.run(opt, feed_dict={x: input, y: output})

    # test
    prediction_value = session.run(prediction, feed_dict={x: input})

    # show
    plt.figure()
    plt.scatter(input, output)
    plt.plot(input, prediction_value, 'r-', lw=5)  # 曲线是预测值
    plt.show()
