import tensorflow.compat.v1 as tf

tf.disable_eager_execution()  # 关闭eager运算
tf.disable_v2_behavior()  # 禁用TensorFlow 2.x行为
import numpy as np
import matplotlib.pyplot as plt

# 使用numpy初始化生成x值
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # np.newaxis 新增一维，从(200)->(200,1)
# x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
# print(x_data.shape)
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# 占位符placeholder定义变量
x = tf.placeholder(tf.float32, [None, 1])  # shape = [None, 1] 表示列是1，行不定）
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层（变量）
W_L1 = tf.Variable(tf.random_normal([1, 10]))  # 加入权重项 weight
b_L1 = tf.Variable(tf.zeros([1, 10]))  # 加入偏置 bias
i_L1 = tf.matmul(x, W_L1) + b_L1  # L1 input = x * w1 + b1  (200,1) * (1,10) = (200,10)
o_L1 = tf.nn.tanh(i_L1)  # 激活函数激活

# 定义神经网络输出层（变量）
W_L2 = tf.Variable(tf.random_normal([10, 1]))  # 加入权重项 weight
b_L2 = tf.Variable(tf.zeros([1, 1]))  # 加入偏置 bias
i_L2 = tf.matmul(o_L1, W_L2) + b_L2  # L2 input = L1_o * w2 + b2  (200,10) * (10,1) = (200,1)
o_L2 = tf.nn.tanh(i_L2)  # 激活函数激活

# 损失函数,均方差
loss = tf.reduce_mean(tf.square(y - o_L2))
# 反向传播 (梯度下降法)，学习率为0.1，最小损失
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 创建绘画
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # # 使用fetch查看中间参数形状
    # r = sess.run([train_step, W_L1, b_L1, i_L1], feed_dict={x: x_data, y: y_data})
    # print(type(r))
    # print(type(r[0]), type(r[1]), type(r[2]), type(r[3]))
    # print("W_L1,b_L1,i_L1", r[1].shape, r[2].shape, r[3].shape)
    # 输出：
    # <class 'list'>
    # <class 'NoneType'> < class 'numpy.ndarray' > < class 'numpy.ndarray' > < class 'numpy.ndarray' >
    #  W_L1, b_L1, i_L1(1, 10)(1, 10)(200, 10)
    # 训练2000次
    for i in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})

    # 预测
    predict = sess.run(o_L2, feed_dict={x: x_data, y: y_data})
    # 绘图
    plt.figure()
    # 绘制初始值散点图
    plt.scatter(x_data, y_data, c='c')
    # 绘制预测曲线图
    plt.plot(x_data, predict, 'm-', lw=1)
    plt.show()
