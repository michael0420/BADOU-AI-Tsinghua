# 目标：使用keras接口进行手写数字识别
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 准备数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_image.shape = ", train_images.shape)
print("train_labels = ", train_labels)
print("test_images.shape = ", test_images.shape)
print("test_labels = ", test_labels)

# 训练数据预处理
train_images = train_images.reshape((60000, 28 * 28))  # 把图片转换为一维,(60000, 784)
train_images = train_images.astype('float32') / 255 * 0.99 + 0.01  # 归一化
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255 * 0.99 + 0.01
# 准备测试用标签 onehot转换，例如test_lables[0]的值由7转变为数组[0,0,0,0,0,0,0,1,0,0,]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 构建神经网络模型，设置输入，输出节点，激活函e数
network = models.Sequential()  # 模型实例化
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))  # 新增一层隐层，输出512，激活函数用relu，输入格式为（28*28，）
network.add(layers.Dense(10, activation='softmax'))  # 新建一层全连接层输出层，输入为上一层的输出结果，全连接层激活函数使用softmax
# tf.compile:配置模型
# optimizer:优化方法， rmsprop：除学习率可调整外，建议保持优化器的其他默认参数不变，该优化器通常是面对递归神经网络时的一个良好选择
# metrics=["accuracy"] 计算准确率
# 该accuracy就是大家熟知的最朴素的accuracy。比如我们有6个样本，其真实标签y_true为[0, 1, 3, 3, 4, 2]，但被一个模型预测为了[0, 1, 3, 4, 4, 4]，即y_pred=[0, 1, 3, 4, 4, 4]，那么该模型的accuracy=4/6=66.67%。
# softmax 的交叉熵损失函数 如果数据的分类标签是 数字 0 1 0 2 2 0 这种 就用sparse_categorical_crossentropy
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
# network.fit(train_images, train_labels, epochs=10, batch_size=128)
network.fit(train_images, train_labels, epochs=20, batch_size=784)

# 评估模型（测试集验证）
# verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar.
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss', test_loss)
print('test_acc', test_acc)

# 输入一张手写数字图片，验证识别效果
x = 15
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
img = test_images[x].reshape(28, 28)
plt.imshow(img, cmap=plt.cm.binary)
plt.show()
# 预测所有测试数据图像的结果
result = network.predict(test_images)
print('图片对应的数字是', np.argmax(result[x]))  # 预测结果的第x张图，最大值的索引
