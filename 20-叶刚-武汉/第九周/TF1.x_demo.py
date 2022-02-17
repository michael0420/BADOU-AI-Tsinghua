"""
tensorflow 1.x版本简单的示例
使用的版本：TensorFlow-gpu=1.15
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 去除“Tensorflow 1.X : tf.xx is deprecated, Please use tf.compat.v1.xx instead”的警告
# 设置logging到ERROR级别，Warning就不会输出了
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# 1.数据准备，使用numpy生成一些数据
# np.linspace(-0.5, 0.5, 200)在-0.5和0.5之间返回均匀间隔的200个数据，即等差数列，列表形式
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]     # (200, 1)
print('x_data.shape = ', x_data.shape)
noise = np.random.normal(0, 0.02, x_data.shape)     # 满足正态分布的随机数
y_data = np.square(x_data) + noise  # y = x * x + b     # (200, 1)

# 2.构建神经网络计算图
# 2.1 定义两个占位符placeholder存放输入数据
x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')     # (200, 1)
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')     # (200, 1)

# 2.2 定义神经网络中间层
# tf.Variable()定义图变量，并使用参数进行初始化
weights_l1 = tf.Variable(initial_value=tf.random_normal([1, 10]))   # (1, 10)，中间层10个神经元
biases_l1 = tf.Variable(initial_value=tf.zeros([1, 10]))            # (1, 10)
wx_plus_b_l1 = tf.matmul(x, weights_l1) + biases_l1    # x * w + b  # (200, 10)
output_l1 = tf.nn.tanh(wx_plus_b_l1)  # 加入tanh激活函数

# 2.3 定义神经网络输出层
weights_l2 = tf.Variable(initial_value=tf.random_normal([10, 1]))   # 输出层1个神经元
biases_l2 = tf.Variable(initial_value=tf.zeros([1, 1]))
wx_plus_b_l2 = tf.matmul(output_l1, weights_l2) + biases_l2
y_predict = tf.nn.tanh(wx_plus_b_l2)

# 2.4 定义损失函数（均方差函数）
loss = tf.losses.mean_squared_error(y, y_predict)
# 或者：loss = tf.reduce_mean(tf.square(y - y_predict))

# 2.5 定义反向传播算法优化器（使用梯度下降算法训练）
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
"""
optimizer.minimize() 函数处理了梯度计算和参数更新两个操作
optimizer.compute_gradients() 函数用于获取梯度
optimizer.apply_gradients() 用于更新参数
"""
train_step = optimizer.minimize(loss)

# 3 创建会话，并在会话中执行计算
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 迭代训练
    for epoch in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    # 训练结束，对数据进行预测得到预测结果
    predict_values = sess.run(y_predict, feed_dict={x: x_data})

    """
    # tensorboard查看数据流图
    1.添加代码 writer = tf.summary.FileWriter(logdir='D:/tf-log', graph=sess.graph)
    2.执行结束后，打开Anaconda Prompt，切换到相应的虚拟环境
    3.输入命令 tensorboard --logdir=D:/tf-log 回车
    4.将Anaconda Prompt上打印的地址（如：http://localhost:6006/）复制，粘贴到浏览器中打开，即可查看数据流图
    """
    writer = tf.summary.FileWriter(logdir='D:/tf-log', graph=sess.graph)

    # 画图展示预测结果
    plt.figure()
    plt.scatter(x_data, y_data)  # 散点是真实值
    plt.plot(x_data, predict_values, 'r-', lw=2)  # 曲线是预测值
    plt.show()
