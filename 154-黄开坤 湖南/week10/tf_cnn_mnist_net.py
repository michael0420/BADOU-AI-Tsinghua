#coding:utf-8

'''CNN
https://blog.csdn.net/cuiran/article/details/86604935
https://blog.csdn.net/u010327061/article/details/84078583
'''
# 1.训练数据集
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)    #产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')     # padding='SAME'时情况比较特殊，注意！，这里还要分为两种情况：stride=1和stride>1，1时输出shape不变,补了两圈

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")  #池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，
    # 因为我们不想在batch和channels上做池化，所以这两个维度设为了1."SAME"是对剩余的不够补零((inputshape-ksize + 2*pad)/stride) + 1。一圈没补

# 第一层卷积
w_conv1 = weight_variable([5, 5, 1, 32])    #[kernel_size, input_channel, output_channel]
b_conv1 = bias_variable([32])
#https://blog.csdn.net/agent_snail/article/details/105700777
x_image = tf.reshape(x, [-1, 28, 28, 1])    #-1根据剩余维度计算，从右往左(矩阵) [n,h,w,c]

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)    #[-1, 28, 28, 32]
# print("h_conv1.shape:", h_conv1.shape)  # h_conv1.shape: (?, 28, 28, 32)
h_pool1 = max_pool_2x2(h_conv1)         #[-1, 14, 14, 32]
# print("h_pool1.shape:", h_pool1.shape)  # h_pool1.shape: (?, 14, 14, 32)
# 第二层卷积
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
# print("h_conv2.shape:", h_conv2.shape)  # h_conv2.shape: (?, 14, 14, 64)
h_pool2 = max_pool_2x2(h_conv2)         # h_pool2.shape: (?, 7, 7, 64)
# print('h_pool2.shape:', h_pool2.shape)
# 第一层全连接
w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# print("h_fc1.shape:", h_fc1.shape)      # h_fc1.shape: (?, 1024)
#参数keep_prob: 设置神经元被选中的概率,在初始化时keep_prob是一个占位符, keep_prob = tf.placeholder(tf.float32)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# 全连接第二层
w_fc2 = weight_variable([1024, 10])
b_fc2= bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
# print('shape:', y_conv.shape)       # shape: (?, 10)

# 交叉熵，多分类公式
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_conv*tf.log(y_),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)   #minimize通过更新 var_list 添加操作以最大限度地最小化 loss
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))   #行维度找最大
# print("correct_prediction:", correct_prediction)    #Tensor("Equal:0", shape=(?,), dtype=bool)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))   #bool型转变成float

saver = tf.train.Saver()
# 训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

    saver.save(sess, 'WModel/model.ckpt')

    print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))    # (10000, 784), (10000, 10)shape送入
