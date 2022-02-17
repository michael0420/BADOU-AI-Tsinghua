import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


n_hidden = 10
x_data = np.linspace(-1,1,200)[:,np.newaxis]
noise_data = np.random.normal(0, 0.05,x_data.shape)
y_data = np.square(x_data) + noise_data

x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

w1 = tf.Variable(tf.random_normal([1, n_hidden]))
b1 = tf.Variable(tf.zeros([1, n_hidden]))
z1 = tf.matmul(x,w1) + b1
a1 = tf.nn.tanh(z1)

w2 = tf.Variable(tf.random_normal([n_hidden, 1]))
b2 = tf.Variable(tf.zeros([1]))
z2 = tf.matmul(a1,w2) + b2
prediction = tf.nn.tanh(z2)

loss = tf.reduce_mean(tf.square(prediction - y_data))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

session = tf.Session()
init = tf.global_variables_initializer()

session.run(init)
loss_list = []
for i in range(1000):
	loss_tem = session.run(loss, feed_dict={x:x_data})
	loss_list.append(loss_tem)
	print("loss=",loss_tem)
	session.run(train_step, feed_dict={x:x_data,y:y_data})
	
y_pred = session.run(prediction, feed_dict={x:x_data})

plt.figure()
plt.scatter(x_data,y_data,c="b", marker="^")
plt.plot(x_data,y_pred,"r-", lw = 2)

plt.figure()
plt.plot(loss_list)
plt.show()
session.close()











