import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = tf.sqrt(2. / in_dim)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


G_W1 = tf.Variable(xavier_init(size=[100, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init(size=[128, 784]))
G_b2 = tf.Variable(tf.zeros(shape=[784]))

theta_G = [G_W1, G_W2, G_b1, G_b2]

# 判别模型的输入和参数初始化

D_W1 = tf.Variable(xavier_init(size=[784, 128]))
D_b1 = tf.Variable(tf.zeros(shape=[128]))

D_W2 = tf.Variable(xavier_init(size=[128, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]

"""
随机噪声产生
"""


def sample_z(m, n):
    return np.random.uniform(-1.0, 1.0, size=[m, n])


"""
生成模型：产生数据
"""


def generator(z):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


"""
判别模型:真实值和概率值
"""


def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit

# 读入数据
mnist = input_data.read_data_sets('./data', one_hot=True)
# print(mnist)

Z = tf.placeholder(tf.float32, shape=[None, 100])

X = tf.placeholder(tf.float32, shape=[None, 784])
# 喂入数据
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)
# 计算loss
D_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real,
                                                                     labels=tf.ones_like(D_logit_real)))
D_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                     labels=tf.zeros_like(D_logit_fake)))
D_loss = D_fake_loss + D_real_loss

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake,
                                                                labels=tf.ones_like(D_logit_fake)))

D_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list= theta_D)
G_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list= theta_G)

if not os.path.exists('out/'):
    os.makedirs('out/')
"""
画图
"""


def plot(samples):
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=.05)
    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')


print("=====================开始训练============================")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for it in range(100000):
        X_mb, _ = mnist.train.next_batch(batch_size=128)
        # print(X_mb)
        _, D_loss_curr = sess.run([D_optimizer, D_loss],
                                  feed_dict={X: X_mb, Z: sample_z(128, 100)})
        _, G_loss_curr = sess.run([G_optimizer, G_loss],
                                  feed_dict={Z: sample_z(128, 100)})
        if it % 1000 == 0:
            print('====================打印出生成的数据============================')
            samples = sess.run(G_sample, feed_dict={Z: sample_z(16, 100)})
            plot(samples)
            plt.show()
        if it % 1000 == 0:
            print('iter={}'.format(it))
            print('D_loss={}'.format(D_loss_curr))
            print('G_loss={}'.format(G_loss_curr))