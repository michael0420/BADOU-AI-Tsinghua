# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/3/27

import random
import cv2
import numpy as np
from copy import copy

def GaussNoise(src, mean, sigma, percentage = 0.8):
    img_noise = copy(src)
    '''
    需要使用显式拷贝，如果直接用赋值符号‘=’，即img_noise = src，会导致src跟着img_noise变化，
    因为赋值的话src与img_noise的内存地址相同，变成了引用传递。
    '''
    m = src.shape[0]
    n = src.shape[1]
    num_noise = int(percentage * m * n)
    for i in range(num_noise):   # range()的参数要求是整数，因此num_noise必须使用类型强制转换
        randX = random.randint(1, m-1)  # 图像边缘不做处理
        randY = random.randint(1, n-1)
        '''
        高斯噪声Pout = Pin + Gauss_noise，其中要加高斯噪声的像素位置是随机的，高斯噪声的值也是随机的
        因此，必须先用函数产生随机坐标，然后用函数产生随机像素值。
        所谓高斯噪声，就是随机产生的高斯噪声大小整体成高斯分布，但会加到随机产生的区域
        '''
        img_noise[randX, randY] = img_noise[randX, randY] + random.gauss(mean, sigma)
        '''
        random.gauss(mean, sigma)得到一个浮点型数据，但与img_noise[randX, randY]相加后赋给img_noise[randX, randY]会自动转换为整型数据
        '''
        if img_noise[randX, randY] < 0:  # 得到的数据可能会超出边界[0, 255]，因此需要裁切
            img_noise[randX, randY] = 0
        elif img_noise[randX, randY] > 255:
            img_noise[randX, randY] = 255

    return img_noise

img = cv2.imread('lenna.png', 0)
mean = 2
sigma = 4
percentage = 0.8
dst = GaussNoise(img, mean, sigma, percentage)
cv2.imshow('src image', img)
cv2.imshow('dst image', dst)
cv2.waitKey(0)


'''
# 下面几行代码的错误在于没有对像素位置进行随机选取，而是按照顺序一一加上高斯噪声,另外没有进行像素值裁切
img = cv2.imread('lenna.png', 0)
mean = 20
sigma = 10
m = img.shape[0]
n = img.shape[1]
GaussNoise = np.zeros([m, n])
for i in range(m):
    for j in range(n):
        GaussNoise[i, j] = random.gauss(mean, sigma)
print('GaussNoise:\n', GaussNoise)

dst = img + GaussNoise

dst = np.uint8(dst)
cv2.imshow('dst image', dst)
cv2.imshow('src image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''