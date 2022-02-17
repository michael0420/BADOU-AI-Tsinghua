import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""
a = np.array([1, 2, 3])
print(a[:, np.newaxis]) 
>> [[1]
    [2]
    [3]]
"""
print(tf.__version__)

#等差数据点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis] #由一维转成二维
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

#定义两个placeholder存放数据
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

#定义神经网络中间层
weight_l1 = tf.Variable(tf.random_normal([1, 10]))
bias_l1 = tf.Variable(tf.zeros([1, 10]))
wx_plus_bias_l1 = tf.matmul(x, weight_l1) + bias_l1 #weight * x + bias
L1 = tf.nn.tanh(wx_plus_bias_l1) #i激活函数

#定义神经网络输出层
weight_l2 = tf.Variable(tf.random.normal([10, 1]))
bias_l2 = tf.Variable(tf.zeros([1, 1]))
wx_plus_bias_l2 = tf.matmul(L1, weight_l2) + bias_l2
prediction = tf.nn.tanh(wx_plus_bias_l2)

#定义损失函数(均方差）
loss = tf.reduce_mean(tf.square(y - prediction))
#定义反向传播算法（梯度下降）
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    #训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x:x_data, y:y_data})

    #获得预测值
    prediction_value = sess.run(prediction, feed_dict={x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)   #散点是真实值
    plt.plot(x_data,prediction_value, 'r-', lw=5)   #曲线是预测值
    plt.show()