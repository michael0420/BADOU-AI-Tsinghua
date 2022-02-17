'''
@author:Jelly
用于计算图片直方图
图像的灰度直方图就描述了图像中灰度分布情况

相关接口：
def Hist_Self(img,channel,histSize,draw=False):
函数用途：计算输入图像的直方图
参数：img：输入的图像
     channel：需要作直方图的维度（通道数）
     GraySize：直方图大小,每个通道色域值
     draw：绘制直方图
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

        '''
        For循环实现，各个色阈值的个数
        n = 0 #记录当前要统计的色域值
        while n<histSize:
            for j in range(h):
                for k in range(w):
                    if(img[j,k,i]==n):
                        sum = sum + 1
            img_hist[n,i] = sum
            n = n + 1
            sum = 0
        n = 0
        '''
        '''
        使用list.count统计所有元素出现的个数
        '''
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

###################################################
img = cv2.imread('lenna.png') #BGR
print('前三张为自己写的绘制多通道直方图')
img_hist = Hist_Self(img,3,256,draw=True)
####################################################



####################################################
print('绘制灰度直方图')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure()
plt.hist(img_gray.ravel(),bins=256)
plt.show()

img_gary_hist = Hist_Self(img_gray,1,256,True)
######################################################