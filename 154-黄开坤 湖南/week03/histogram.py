#coding:utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

'''直方图获取'''


# 灰度图像-->直方图
# 1.方式一(柱状图)
img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #(512, 512)
#
# plt.figure()        # 创建自定义图像
# plt.hist(img_gray.ravel(), 256)  #直方图。plt.hist(src,pixels) src:数据源，注意这里只能传入一维数组，使用src.ravel()可以将二维图像拉平为一维数组。pixels:像素级，一般输入256。
# plt.show()

'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''

# 方式二
hist = cv2.calcHist([img_gray], [0], None, [256], [0,256])  #计算灰度级别出现频数(数量)，返回的是一个（256,1）的数组
plt.figure()
plt.title('Gray_image to Histogram')
plt.xlabel('Bins')      # x轴标签
plt.ylabel('# of Pixels')   #Y轴标签
plt.plot(hist)      #用于画图,它可以绘制点和线
plt.xlim([0, 256])  #设置x坐标轴范围
plt.show()


'''彩色直方图'''
# image = cv2.imread('lenna.png')
# # cv2.imshow('Original', image)
# # cv2.waitKey(0)
#
# chans = cv2.split(image)
# # print(chans)
# colors = ('b', 'g', 'r')
# plt.figure()
# plt.title('Colors of histogram')
# plt.xlabel('Bins')
# plt.ylabel('# of Pixels')
#
# for (chan, col) in zip(chans, colors):  #zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
#     hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
#     plt.plot(hist, color=col)
#     plt.xlim([0, 256])
# plt.show()


'''获取全局直方图, 搜索, 调用cv2.calcHist函数'''
# 获得上图的直方图，首先得先进行灰度化，定义一个函数
# def image_calcuhist(img_path):
#     img = cv2.imread(img_path)      # 读取图片
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   ## 灰度化
#     hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]) # 计算灰度级别出现频率，返回的是一个（256,1）的数组
#     for i in range(hist.shape[0]):
#         hist[i] = hist[i]/(img_gray.shape[0] * img_gray.shape[1])       # 对获得的直方图数据进行归一化,将数组中的值替换掉
#     return hist
#
# if __name__ == '__main__':
#     x = np.linspace(0, 256, 256)    #产生从start到stop的等差数列，num为元素个数，默认50个.# 横坐标灰度级别
#     y = image_calcuhist('lenna.png')    # 纵坐标值获取
#     plt.bar(x, y.ravel(), 0.9, alpha=1, color='b')  ## 通过matplotlib进行直方图的绘制,alpha:scalar or None
#     plt.show()


'''自定义实现'''
# 自定义函数进行灰度直方图的绘制
# def image_histdefinition(imagepath):
#     img = cv2.imread(imagepath)  # 读取图片
#     imgInfo = img.shape  # 获得图片的尺寸大小
#     height = imgInfo[0]
#     width = imgInfo[1]
#     # 灰度化
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     count = np.zeros(256, np.float)  # 共256个，创建一个数组用于存放灰度级别出现的频率
#     # 计算灰度级别频率
#     for i in range(0, height):
#         for j in range(0, width):
#             pixel = gray[i, j]  # 获取灰度等级
#             index = int(pixel)  # 强制类型转换
#             count[index] = count[index] + 1
#     # 计算出现概率，即归一化
#     for i in range(0, 255):
#         count[i] = count[i] / (height * width)
#     return count
#
#
# if __name__ == '__main__':
#     # 绘图
#     x = np.linspace(0, 255, 256)
#     count = image_histdefinition('lenna.png')
#     y = count
#     plt.bar(x, y, 0.9, alpha=1, color='b')
#     plt.show()
