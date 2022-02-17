import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 构建图
def constrcution():  # 构造图
    matrix1 = tf.constant([[3., 3.]])  # op1
    print(matrix1)
    matrix2 = tf.constant([[2.], [2.]])  # op2
    print(matrix2)
    product = tf.matmul(matrix1, matrix2)  # op3
    print(np.array(product))
    # 启动默认图
    Sess = tf.Session()
    result = Sess.run(product)  # 执行op3
    print(result)
    # 任务完成，关闭对话
    Sess.close()


def Fetch():
    input1 = tf.constant(3.0)
    input2 = tf.constant(5.0)
    input3 = tf.constant(2.0)
    addsum = tf.add(input2, input3)
    multisum = tf.multiply(input1, addsum)
    Sess = tf.Session()
    result = Sess.run([multisum, addsum])  # fetch操作，获取执行节点的过程值
    print(result)
    Sess.close()


def Feed():
    input1 = tf.placeholder(tf.float32)  # placeholder 占位
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)
    Sess = tf.Session()
    result = Sess.run([output], feed_dict={input1: [2.], input2: [7.]})  # feed操作,进行替换
    print(result)
    Sess.close()

def NN():
    #  [:, np.newaxis] 插入新的维度
    x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]   # 生成200个-0.5~0.5的随机点
    noise = np.random.normal(0,0.02,x_data.shape)
    y_data = np.square(x_data)+noise

    # 定义两个placeholder存放输入数据
    x = tf.placeholder(tf.float32,[None,1])   # shape 为 1
    y = tf.placeholder(tf.float32,[None,1])

    # 定义神经网路隐藏层
    Weights_L1 = tf.Variable(tf.random_normal([1,10]))
    bias_L1 = tf.Variable(tf.zeros([1,10]))  # 偏置项
    Wx_plus_b_L1 = tf.matmul(x,Weights_L1)+bias_L1
    L1 = tf.nn.tanh(Wx_plus_b_L1)  # 通过激活函数输出

    # 定义神经网络输出层
    Weights_L2 = tf.Variable(tf.random_normal([10,1]))
    bias_L2 = tf.Variable(tf.zeros([1,1]))   # 加入偏置项
    Wx_plus_b_L2 = tf.matmul(L1,Weights_L2)+bias_L2
    prediction = tf.nn.tanh(Wx_plus_b_L2)  # 通过激活函数输出

    # 定义损失函数，此处采用均方差MSE
    loss = tf.reduce_mean(tf.square(y-prediction))
    # 定义反向传播算法(使用梯度下降法训练)、
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.Session() as sess:
        # 变量初始化
        sess.run(tf.global_variables_initializer())
        # 训练2000次
        for i in range(20000):
            sess.run(train_step,feed_dict={x:x_data,y:y_data})
        # 获得预测值
        prediction_value=sess.run(prediction,feed_dict={x:x_data})
        plt.figure()
        plt.scatter(x_data,y_data)
        plt.plot(x_data,prediction_value,'r-',lw=5)  # 曲线为预测值
        plt.show()
constrcution()
Fetch()
Feed()
NN()