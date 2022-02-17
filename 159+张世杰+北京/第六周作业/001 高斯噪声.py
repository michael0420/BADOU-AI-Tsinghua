# encoding: utf-8
import numpy as np
import cv2
from numpy import shape
import random

'''定义高斯噪声函数：原图片，均值，方差，存在噪声像素和总像素比'''


def GauessionNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        '''获取随机坐标位置：边缘位置不包括'''
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        '''添加高斯噪声：为原图像添加服从高斯分布的随机数，确定：有可能存在覆盖已经添加过噪声的像素点， 最终噪声的像素点个数少于NoiseNum'''
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        '''截断操作：异常数据处理'''
        if NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('lenna.png', 0)
img1 = GauessionNoise(img, 2, 4, 1)
img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source', img2)
cv2.imshow('lenna_GaussianNoise', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
