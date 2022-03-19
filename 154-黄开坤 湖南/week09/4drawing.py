#coding:utf-8

import numpy as np
import matplotlib.pyplot as plt

#open函数里的路径根据数据存储的路径来设置
data_file = open('dataset/mnist_test.csv')
data_list = data_file.readlines()   #readlines()读取所有行然后把它们作为一个字符串列表返回
data_file.close()
print(len(data_list))
print(data_list[0])

#把数据依靠‘,’区分，并分别读入
all_values = data_list[0].split(',')
print('all_values:', all_values)
#第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = np.asfarray(all_values[1:]).reshape((28, 28)) #asfarray()把普通数组转换成浮点型数组
plt.imshow(image_array, cmap='Greys', interpolation='None')
plt.show()

#数据预处理（归一化）
scaled_input = image_array / 255.0 * 0.99 + 0.01
print(scaled_input, scaled_input.shape)
'''
从绘制的结果看，数据代表的确实是一个黑白图片的手写数字。
数据读取完毕后，我们再对数据格式做些调整，以便输入到神经网络中进行分析。
我们需要做的是将数据“归一化”，也就是把所有数值全部转换到0.01到1.0之间。
由于表示图片的二维数组中，每个数大小不超过255，由此我们只要把所有数组除以255，就能让数据全部落入到0和1之间。
有些数值很小，除以255后会变为0，这样会导致链路权重更新出问题。
所以我们需要把除以255后的结果先乘以0.99，然后再加上0.01，这样所有数据就处于0.01到1之间。
'''