#coding:utf-8

'''
tf下的MNIST
https://blog.csdn.net/sinat_34328764/article/details/83832487
'''

# 二层网络
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# #
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#
# trainimg  = mnist.train.images
# trainlabel = mnist.train.labels
# testimg = mnist.test.images
# testlabel = mnist.test.labels
#
# #placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
# x = tf.placeholder(tf.float32, [None, 784])
# y = tf.placeholder(tf.float32, [None, 10])
# # 图变量
# w = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
#
# actv = tf.nn.softmax(tf.matmul(x, w) + b)
# # 链接。多分类公式
# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(actv, 1e-10, 1.0)), reduction_indices=1))  #reduction_indices=1横向对矩阵求和
#
# learningrate = 0.01
# optm = tf.train.GradientDescentOptimizer(learningrate).minimize(cost)   # 梯度下降
# pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))    # 数组行维度找最大，相等比较
# accr = tf.reduce_mean(tf.cast(pred, tf.float32))
#
# init_op = tf.global_variables_initializer() # 初始化
#
# epochs = 50
# batch_size = 100
# display_step = 5    # 用来比较、输出结果
#
# #启动图，运行op
# with tf.Session() as sess:
#     sess.run(init_op)   #对变量进行初始化，真正的赋值操作
#     #
#     for epoch in range(epochs):
#         avg_cost = 0
#         num_batch = int(mnist.train.num_examples / batch_size)  # 多少个batch
#         #
#         for i in range(num_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size) ##其中的batch_size代表返回多少个训练数据集和对应的标签
#             # 运行模型进行训练
#             sess.run(optm, feed_dict={x: batch_xs, y:batch_ys}) # feed_dict(x:, y:)喂数据
#             feeds = {x:batch_xs, y:batch_ys}
#             # 如果觉得上面 feed_dict 的不方便 也可以提前写在外边
#             avg_cost += sess.run(cost, feed_dict=feeds) / num_batch
#
#         if epoch % display_step ==0:
#             feed_train = {x: trainimg, y:trainlabel}
#             feed_test = {x: mnist.test.images, y:mnist.test.labels}
#             train_acc = sess.run(accr, feed_dict=feed_train)
#             test_acc = sess.run (accr, feed_dict=feed_test)
#             print("Eppoch: %03d/%03d cost: %.9f train_acc: %.3f test_acc: %.3f" %
#                   (epoch, epochs, avg_cost, train_acc, test_acc))
# print("Done.")


# # 三层网络, 失败
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
#
# #
# mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#
# trainimg  = mnist.train.images
# trainlabel = mnist.train.labels
# testimg = mnist.test.images
# testlabel = mnist.test.labels
#
# #placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
# x = tf.placeholder(tf.float32, [None, 784])
# # x1 = tf.placeholder(tf.float32, [784, 200])
# y1 =tf.placeholder(tf.float32, [784, 200])
# y2 = tf.placeholder(tf.float32, [None, 10])
# # 图变量
# w1 = tf.Variable(tf.zeros([784, 200]))
# w2 = tf.Variable(tf.zeros([200, 10]))
# b1 = tf.Variable(tf.zeros([200]))
# b2 = tf.Variable(tf.zeros([10]))
#
# actv1 = tf.nn.relu(tf.matmul(x, w1) + b1)
# actv2 = tf.nn.softmax(tf.matmul(actv1, w2) + b2)
# # loss
# cost = tf.nn.softmax_cross_entropy_with_logits(logits=actv2, labels=y2)
# print("cost:", cost)
#
# learningrate = 0.01
# optm = tf.train.GradientDescentOptimizer(learningrate).minimize(cost)   # 梯度下降
# pred = tf.equal(tf.argmax(actv2, 1), tf.argmax(y2, 1))    # 数组行维度找最大，相等比较
# print("pred:", pred)
# accr = tf.reduce_mean(tf.cast(pred, tf.float32))
# print('accr:', accr)
#
# init_op = tf.global_variables_initializer() # 初始化
#
# epochs = 50
# batch_size = 100
# display_step = 5    # 用来比较、输出结果
#
# #启动图，运行op
# with tf.Session() as sess:
#     sess.run(init_op)   #对变量进行初始化，真正的赋值操作
#     #
#     for epoch in range(epochs):
#         avg_cost = 0
#         num_batch = int(mnist.train.num_examples / batch_size)  # 多少个batch
#         #
#         for _ in range(num_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size) ##其中的batch_size代表返回多少个训练数据集和对应的标签
#             # 运行模型进行训练
#             sess.run(optm, feed_dict={x: batch_xs, y2:batch_ys}) # feed_dict(x:, y:)喂数据
#             feeds = {x:batch_xs, y2:batch_ys}
#             # 如果觉得上面 feed_dict 的不方便 也可以提前写在外边
#             avg_cost += sess.run(cost, feed_dict=feeds) / num_batch
#
#         if epoch % display_step ==0:
#             feed_train = {x: trainimg, y2:trainlabel}
#             feed_test = {x: testimg, y2:testlabel}
#             train_acc = sess.run(accr, feed_dict=feed_train)
#             print('train_acc:', train_acc)
#             test_acc = sess.run (accr, feed_dict=feed_test)
#             print("test_acc:", test_acc)
#             print("Eppoch: %03d/%03d cost:  train_acc: %.3f test_acc: %.3f" %
#                   (epoch, epochs, train_acc, test_acc))
# print("Done.")

