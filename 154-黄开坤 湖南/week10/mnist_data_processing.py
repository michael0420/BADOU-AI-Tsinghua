#coding:utf-8

'''数据处理，了解
https://blog.csdn.net/weixin_41055137/article/details/81071226?utm_source=copy
'''

from keras.datasets import mnist
import matplotlib.pyplot as plt

'''
所以这里返回的是手写图片的两个tuple，第一个tuple存储的是我们已经人工分类好的图片，也就是每一张图片都有自己对应的标签，然后可以拿来训练，
第二个tuple存储的是我们还没分类的图片，在第一个tuple训练完后，我们可以把第二个tuple利用神经网络进行分类，
根据实验结果的真实值与我们的预测值进行对比得到相应的损失值，再利用反向传播进行参数更新，再进行分类，然后重复前述步骤直至损失值最小
'''
# load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_shape:', x_train.shape)    # x_shape: (60000, 28, 28)
print('y_shape:', y_train.shape)    # y_shape: (60000,)
print('x_test:', x_test.shape)      # (10000, 28, 28)
print('y_test:', y_test.shape)      # (10000, )

from keras.utils import np_utils
print(y_test.shape[0])      # 1000
y_test = np_utils.to_categorical(y_test)
print('y_test.shape:', y_test.shape)



# plt.subplot(331)    #这个subplot函数的作用是确定图像的位置以及图像的个数，前两个3的意思是可以放9张图片，如果变成221的话，就是可以放4张图片，然后后面的1，是确定图像的位置，处于第一个，以下的subplot同理
# plt.imshow(x_train[0], cmap=plt.get_cmap(('gray'))) #X_train存储的是图像的像素点组成的list数据类型，这里面又由一个二维的list（28 x 28的像素点值）和一个对应的标签list组成，y_train存储的是对应图像的标签，也就是该图像代表什么数字
# plt.subplot(332)
# plt.imshow(x_train[1], cmap=plt.get_cmap('gray'))
# plt.subplot(333)
# plt.imshow(x_train[2], cmap=plt.get_cmap('gray'))
# plt.subplot(334)
# plt.imshow(x_train[3], cmap=plt.get_cmap('gray'))
# plt.subplot(335)
# plt.imshow(x_train[4], cmap=plt.get_cmap('gray'))
# plt.subplot(336)
# plt.imshow(x_train[5], cmap=plt.get_cmap('gray'))
# plt.subplot(337)
# plt.imshow(x_train[6], cmap=plt.get_cmap('gray'))
# plt.subplot(338)
# plt.imshow(x_train[7], cmap=plt.get_cmap('gray'))
# plt.subplot(339)
# plt.imshow(x_train[8], cmap=plt.get_cmap('gray'))
# plt.show()
