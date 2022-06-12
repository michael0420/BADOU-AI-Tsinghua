from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 1. 使用tf.keras.datasets.mnist.load_data()函数下载mnist手写数字训练数据和检测数据加载到内存中(第一次运行需要下载数据，会比较慢)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
"""
load_data() returns tuple of NumPy arrays : (x_train, y_train), (x_test, y_test).
train_images: uint8 NumPy array of grayscale image data with shapes (60000, 28, 28), containing the training data. 
Pixel values range from 0 to 255.
train_labels: uint8 NumPy array of digit labels (integers in range 0-9) with shape (60000,) for the training data.
test_images: uint8 NumPy array of grayscale image data with shapes (10000, 28, 28), containing the test data. 
Pixel values range from 0 to 255.
test_labels: uint8 NumPy array of digit labels (integers in range 0-9) with shape (10000,) for the test data.
"""
# 使用断言关键字验证
assert train_images.shape == (60000, 28, 28)
assert test_images.shape == (10000, 28, 28)
assert train_labels.shape == (60000,)
assert test_labels.shape == (10000,)
print('tran_labels:', train_labels)  # 打印结果表明，第一张手写数字图片的内容是数字5，第二种图片是数字0，以此类推。
print('test_labels:', test_labels)  # 输出结果表明，用于检测的第一张图片内容是数字7，第二张是数字2，依次类推。

# 2. 显示测试集的第一张图片(非必需步骤)
digit = test_images[0]

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 3. 搭建神经网络
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()  # 创建实例命名为network。Sequential意为顺序的，即序贯模型，它是多个网络层的线性堆叠，也就是“一条路走到黑”。
# 下述layers.Dense层的第一个参数units(节点数)是自行设定
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))  # 第一层需要加一个input_shape关键字参数，参数不包含batch大小
network.add(layers.Dense(10, activation='softmax'))  # 输出层的units=10，意味着存在10个类别，实际意义为输出的结果是从0~10这是个数字。

network.summary()  # 输出一下搭建的神经网络框架总结

# 在训练模型之前，我们需要通过compile来对学习过程进行配置。compile接收三个参数：优化器optimizer、损失函数loss和指标列表metrics。
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. 对训练集和测试集的数据做预处理
# 4.1 将每张图片数据由(28， 28)的二维数组变成(28 * 28)的一维数组
# 4.2 同时将灰度图中[0, 255]的整数灰度值归一化为[0,1]的浮点数
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# 5. 对训练集和测试集的标签数据做预处理
# 对应输出层的units=10，将标签做one-hot编码，用一个拥有10个元素的一位数组替换标签。
# 我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0，即[0,0,0,0,0,0,0,1,0,0]
from tensorflow.keras.utils import to_categorical

print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# 6. 开始使用数据对神经网络进行训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)
"""
传入fit()各变量的含义：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算训练数据将会被遍历5次
fit函数返回一个History的对象，其History.history属性记录了损失函数和其他指标的数值随epoch变化的情况，如果有验证集的话，也包含了验证集的这些指标变化情况。
"""

# 7. 测试数据输入，检验网络学习后的图片识别效果. P.S. 识别效果与硬件有关(CPU/GPU)
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
# verbose: 日志记录——0:静默不显示任何信息，1(default):输出进度条记录
print('test_loss', test_loss)  # 打印loss
print('test_acc', test_acc)  # 打印accuracy

# 8. 输入一张手写数字图片到网络中，看看它的识别效果
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]  # 挑选测试集第2张图片，是数字2

plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)  # 应用已经训练好的模型进行预测

for i in range(res[1].shape[0]):  # 提取第二个预测结果，即对数字2的预测结果
    if res[1][i] == 1:  # 看结果中10个元素的数组中第几位是1
        print("the number for the picture is : ", i)  # 第几位是1就输出这个数字的预测结果是几
        break
