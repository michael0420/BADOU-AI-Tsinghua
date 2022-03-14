import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#使用np生成200随机点
xData=np.linspace(start=-0.5,stop=0.5,num=200)[:,np.newaxis]
#横向变纵向
print(xData.shape)
noise=np.random.normal(loc=0,scale=0.02,size=xData.shape)
yData=np.square(xData)+noise
#建立y=x^2函数并加上噪音

#定义两个placeholder存放输入数据
#shape：数据形状。默认是None，就是一维值，
# 也可以是多维,比如[2,3], [None, 3]表示列是3，行不定）
x=tf.placeholder(dtype=tf.float32,shape=[None,1])
y=tf.placeholder(dtype=tf.float32,shape=[None,1])

#定义中间层
#权重使用正态初始化，形状为一行10列
weightsL1=tf.Variable(tf.random_normal([1, 20]))
print(weightsL1)
biasesL1=tf.Variable(tf.zeros([1,20]))
outputL1= tf.matmul(x, weightsL1) + biasesL1
print(outputL1)
L1=tf.nn.tanh(outputL1)

#定义输出层
weightsL2=tf.Variable(tf.random_normal([20,1]))
biasesL2=tf.Variable(tf.zeros([1,1]))
outputL2=tf.matmul(L1,weightsL2)+biasesL2
prediction=tf.nn.tanh(outputL2)

#定义损失函数为均方差,
loss=tf.reduce_mean(tf.square(y-prediction))
#使用学习率为0.1的反向传播算法最小化loss
trainStep=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #全局变量初始化
    for i in range(1000):
        sess.run(trainStep,feed_dict={x:xData,y:yData})

    predictValues=sess.run(prediction,feed_dict={x:xData})

    plt.figure()
    plt.scatter(xData,yData)
    plt.plot(xData,predictValues,'r-',lw=4)
    plt.show()