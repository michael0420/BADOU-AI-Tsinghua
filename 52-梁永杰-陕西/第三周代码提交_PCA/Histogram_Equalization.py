'''
@auther:Jelly
图片直方图均衡化

相关接口：
    def Hist_Equal()
    函数用途：用于将图片的直方图进行均衡化处理，达到输出较好的输出效果
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

def Hist_Self(img,channel,histSize,draw=False):
    '''
    函数用途：计算输入图像的直方图
    参数：img：输入的图像
         channel：需要作直方图的维度（通道数）
         GraySize：直方图大小,每个通道色域值
         draw：绘制直方图
    '''
    img_hist = np.zeros((histSize,channel))

    if img.ndim<3:    # 防止有单通道图像只有二维，添加一个维度
        img = img[:,:,np.newaxis]

    [h,w,channel] = img.shape

    sum = 0 #记录当前统计的色域值的个数
    for i in range(channel):
        n = 0  # 记录当前要统计的色域值
        #将每个通道的图像矩阵重新排列成一维数组
        # 将numpy数组改为list从而能够使用count方法，进行目标元素个数统计
        img_list = (img[:,:,i].ravel()).tolist()
        while n < histSize:
            img_hist[n,i] = img_list.count(n) #进行统计时只能对一维数组进行统计，其他维度数组均不能进行统计，返回值为0
            n = n + 1
        n = 0

    if draw == True:                          #进行画图设置
        import matplotlib.pyplot as plt
        for i in range(channel):
            plt.figure()
            plt.xlabel("print buy Hist_Self")
            plt.plot(img_hist[:,i])
            plt.hist((img[:,:,i]).ravel(), bins=256)
        plt.show()
    return img_hist


def Hist_Equal(img_D,histSize=256):
    '''
    函数用途：作图片一个维度的直方图均衡化
    参数说明：
            img_D：一个维度的图片数据
            histSize：直方图的像素范围
    '''
    [h, w] = img_D.shape
    img_HistE = np.array(img_D)
    img_Hist = Hist_Self(img_D,1,256,draw=False)

    Size_image = h*w           # 得到图像大小
    sum_Pi = 0                 # 用于累积加和结果
    for pix in range(histSize):
        Pi = float(img_Hist[pix,0])/Size_image
        sum_Pi = sum_Pi + Pi
        q = int(sum_Pi*256-1)
        position = np.where(img_D==pix)
        img_HistE[position] = q

    return img_HistE


img = cv2.imread("lenna.png", 1)
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_equal = img_gray


img_equal = Hist_Equal(img_equal)

plt.figure()
plt.hist(img_gray.ravel(),bins=256)

plt.figure()
plt.xlabel("HIstogram_Equalization")
plt.hist(img_equal.ravel(), bins=256)

plt.figure()
plt.imshow(img_gray,cmap='gray')

plt.figure()
plt.xlabel("HIstogram_Equalization")
plt.imshow(img_equal,cmap='gray')
plt.show()