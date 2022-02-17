"""
    加载keras默认的mnist数据集
    train_images shape =  (60000, 28, 28)
    train_labels shape =  (60000,)
    test_images shape =  (10000, 28, 28)
    test_labels shape =  (60000,)
"""
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_images shape = ", train_images.shape)
print("train_labels shape = ", train_labels.shape)
print("test_images shape = ", test_images.shape)
print("test_labels shape = ", train_labels.shape)


def show_pictures(images, label):
    """
    Args:
        images:显示的图片数据
        labels:标签数据
    Returns:
        显示图像结果
    """
    B, _, _ = images.shape

    plt.figure()
    for index in range(B):
        plt.subplot(1, B, index + 1)
        digit = int(label[index])
        img = images[index]
        plt.imshow(img, cmap='gray')
        plt.title(str(digit))
        plt.xticks([])
        plt.yticks([])
    plt.show()


'''
使用tensorflow.Keras搭建一个有效识别图案的神经网络，
1.layers:表示神经网络中的一个数据处理层。(dense:全连接层)
2.models.Sequential():表示把每一个数据处理层串联起来.
3.layers.Dense(…):构造一个数据处理层。
4.input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组，
后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
'''

from tensorflow.keras import layers
from tensorflow.keras import models

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10,activation='softmax',))

network.compile(optimizer='rmsprop',loss = 'categorical_crossentropy',metrics=['accuracy'])
'''
在把数据输入到网络模型之前，把数据做归一化处理:
1.reshape(60000, 28*28）:train_images数组原来含有60000个元素，每个元素是一个28行，28列的二维数组，
现在把每个二维数组转变为一个含有28*28个元素的一维数组.
2.由于数字图案是一个灰度图，图片中每个像素点值的大小范围在0到255之间.
3.train_images.astype(“float32”)/255 把每个像素点的值从范围0-255转变为范围在0-1之间的浮点值。
'''
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype("float32") / 255

'''
把图片对应的标记也做一个更改：
目前所有图片的数字图案对应的是0到9。
例如test_images[0]对应的是数字7的手写图案，那么其对应的标记test_labels[0]的值就是7。
我们需要把数值7变成一个含有10个元素的数组，然后在第8个元素设置为1，其他元素设置为0。
例如test_lables[0] 的值由7转变为数组[0,0,0,0,0,0,0,1,0,0,] ---one hot
'''

from tensorflow.keras.utils import to_categorical

print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

'''
把数据输入网络进行训练：
train_images：用于训练的手写数字图片；
train_labels：对应的是图片的标记；
batch_size：每次网络从输入的图片数组中随机选取128个作为一组进行计算。
epochs:每次计算的循环是五次
'''

network.fit(train_images,train_labels,batch_size=128,epochs = 5,)

'''
测试数据输入，检验网络学习后的图片识别效果.
识别效果与硬件有关（CPU/GPU）.
'''
digit = test_images[1]
digit = digit.reshape((28,28))
plt.imshow(digit, cmap='gray')
plt.show()
# 返回模型输出结果
# (10000,784) ---> (10000,10)
res = network.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break
# if __name__ == "__main__":
#     # show_pictures(train_images[:4], train_labels[:4])
#     pass