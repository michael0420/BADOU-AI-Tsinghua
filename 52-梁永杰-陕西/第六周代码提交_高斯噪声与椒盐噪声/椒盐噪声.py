'''
@auther:Jelly
@用途：实现高斯噪声
相关函数：
    def S_PNoise_self(img,percetage)
        函数功能：对单通道图像实现添加高斯噪声
        函数参数：
            img：待处理单通道图像
            percetage：模糊像素占比
            返回值：添加高斯噪声后的图像
'''
import numpy as np
import cv2
import random

def S_PNoise_self(img,percetage):
    '''
    函数功能：对单通道图像实现添加高斯噪声
    函数参数：
        img：待处理单通道图像
        percetage：模糊像素占比
        返回值：添加高斯噪声后的图像
    '''
    h, w = img.shape[0], img.shape[1]
    #新建图片
    imgNoise = np.zeros((h,w),dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            imgNoise[i,j] = img[i,j]
    NoiseNum = int(percetage * (h * w))
    for i in range(NoiseNum):
        rand_x = random.randint(0,h-1)
        rand_y = random.randint(0,w-1)
        if random.randint(0,1) >0.5:
            imgNoise[rand_x,rand_y] = 255
        else:
            imgNoise[rand_x,rand_y] = 0
    return imgNoise


img = cv2.imread('lenna.png',0)
img_sp = S_PNoise_self(img,0.2)
cv2.imshow('source',img)
cv2.imshow('lenna_S&PNoise',img_sp)

cv2.waitKey(0)