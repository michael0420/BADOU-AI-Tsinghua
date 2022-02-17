import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings(action='ignore')

x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

W1 = tf.Variable(tf.random.normal([1,10]))
b1 = tf.Variable(tf.zeros([1,10]))
Wb = tf.matmul(x,W1)+b1
L1 = tf.nn.tanh(Wb)

W2 = tf.Variable(tf.random.normal([10,1]))
b2 = tf.Variable(tf.zeros([1,1]))
Wb2 = tf.matmul(L1,W2) + b2
output = tf.nn.tanh(Wb2)

loss = tf.reduce_mean(tf.square(y-output))
Optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        sess.run(Optimizer, feed_dict={x:x_data, y:y_data})
    prediction = sess.run(output,feed_dict={x:x_data})

    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction,'r-',lw=5)
    plt.show()