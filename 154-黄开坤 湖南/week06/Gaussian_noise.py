#coding:utf-8

import cv2 as cv
import random


'''
高斯噪声生成步骤：
1、输入参数：sigma，mean
2、按比例生成高斯随机数
 3、每次取出一个随机点，并加上Random.gauss随机数
 4、将所有像素缩放到[2-255]之间
 5、循环所有高斯随机数的像素
6、输出图像
'''

def GaussinaNoise(src, sigma, mean, percent):
    noiseimg = src
    noisenum = int(percent * src.shape[0] * src.shape[1])
    for i in range(noisenum):
        #随机选择行，列坐标randX, randY
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        #输入 + 高斯随机数
        noiseimg[randX, randY] = noiseimg[randX, randY] + random.gauss(sigma, mean)   #返回：随机高斯分布浮点数
        #放到[0-255]中
        if noiseimg[randX, randY] < 0:
            noiseimg[randX, randY] = 0
        if noiseimg[randX, randY] > 255:
            noiseimg[randX, randY] = 255
    return noiseimg

img = cv.imread('lenna.png', 0)
img1 = GaussinaNoise(img, 2, 4, 0.8)
img = cv.imread('lenna.png')
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('img_gaussian_noise', img1)
cv.imshow('img_raw', img2)
cv.waitKey(0)
cv.destroyWindow('all')
