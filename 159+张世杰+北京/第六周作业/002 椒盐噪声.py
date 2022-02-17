# encoding: utf-8
import numpy as np
import cv2
from numpy import shape
import random

def ps(src,percetage):
    '''定义原图片和教研噪声点的个数'''
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    '''随机寻找NoiseNum个噪声点：可能会重复'''
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        '''根据随机数将像素点的灰度值设为0或255，此时黑白的比例各占一半'''
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    '''返回添加噪声后的图片'''
    return NoiseImg

img=cv2.imread('lenna.png',0)
img1=ps(img,0.2)

img = cv2.imread('lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('source',img2)
cv2.imshow('lenna_PepperandSalt',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()