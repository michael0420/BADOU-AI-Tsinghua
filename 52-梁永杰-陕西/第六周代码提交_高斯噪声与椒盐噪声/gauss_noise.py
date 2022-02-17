'''
@auther:Jelly
@用途：实现高斯噪声
相关函数：
    def GaussNoise_self(img,means,sigma,percetage)
        函数功能：对单通道图像实现添加高斯噪声
        函数参数：
            img：待处理单通道图像
            means：高斯函数均值
            sigma：高斯函数方差
            percetage：模糊像素占比
            返回值：添加高斯噪声后的图像
'''
import numpy as np
import cv2
import random


def GaussNoise_self(img,means,sigma,percetage):
    '''
    函数功能：对单通道图像实现添加高斯噪声
    函数参数：
        img：待处理单通道图像
        means：高斯函数均值
        sigma：高斯函数方差
        percetage：模糊像素占比
        返回值：添加高斯噪声后的图像
    '''
    h,w = img.shape[0],img.shape[1]
    #新建图片
    NoiseImg = np.zeros((h,w),dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            NoiseImg[i,j] = img[i,j]
    NoiseNum = int(percetage * (h * w))

    for i in range(NoiseNum):
        #在Noise个随机点上添加高斯噪声
        rand_x = random.randint(0,h-1)
        rand_y = random.randint(0,w-1)
        NoiseImg[rand_x,rand_y] = img[rand_x,rand_y] + random.gauss(means,sigma)
        #将灰度值限制在0，255内
        if NoiseImg[rand_x,rand_y] < 0:
            NoiseImg[rand_x,rand_y] = 0
        if  NoiseImg[rand_x,rand_y] >255:
            NoiseImg[rand_x,rand_y] = 255
    return NoiseImg


img = cv2.imread('lenna.png',0)
imgGN = GaussNoise_self(img,1,2,0.8)

cv2.imshow('source',img)
cv2.imshow('GassianNoise',imgGN)

cv2.waitKey(0)