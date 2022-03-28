# -*- coding:utf-8 -*-
# author: Damion
# email: 1633245455@qq.com
# creation time: 2022/3/27

import random
import cv2
import numpy as np
from copy import copy

def noise_peppersalt(src, SNR):
    img_noise = copy(src)
    '''
    需要使用显式拷贝，如果直接用赋值符号‘=’，即img_noise = src，会导致src跟着img_noise变化，
    因为赋值的话src与img_noise的内存地址相同，变成了引用传递。
    '''
    m = src.shape[0]
    n = src.shape[1]
    num_noise = int(SNR * m * n)
    for i in range(num_noise):   # range()的参数要求是整数，因此num_noise必须使用类型强制转换
        randX = random.randint(1, m-1)  # 图像边缘不做处理
        randY = random.randint(1, n-1)
        '''
        椒盐噪声跟高斯噪声相同点在于也需要随机产生需要添加噪声的像素坐标，不同的是椒盐噪声一旦确定某个坐标需要添加椒盐噪声
        那么该坐标的值只能是0或255
        '''
        if random.random() < 0.5:
            img_noise[randX, randY] = 0
        elif random.random() >= 0.5:
            img_noise[randX, randY] = 255

    return img_noise

img = cv2.imread('lenna.png', 0)
SNR = 0.4
dst = noise_peppersalt(img, SNR)
cv2.imshow('src image', img)
cv2.imshow('dst image', dst)
cv2.waitKey(0)