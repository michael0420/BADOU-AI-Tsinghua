#coding:utf-8

import random
import cv2 as cv


'''Salt and pepper noise
1、指定信噪比sSNR
2、加噪的像素数目=总的 * SNR
    3、遍历每个要加噪的数目
    4、判断，指定0/255
'''

def SaPep_Noise(src, percent):
    NoiseImg = src
    NoiseNum = int(percent * src.shape[0] * src.shape[1])
    #遍历每个要加噪的像素点
    for i in range(NoiseNum):
        # 生成随机数
        randx = random.randint(0, src.shape[0] - 1)
        randy = random.randint(0, src.shape[1] - 1)
        #判断
        if NoiseImg[randx, randy] <= 0.5:
            NoiseImg[randx, randy] = 0
        else:
            NoiseImg[randx, randy] = 255
    return NoiseImg

img = cv.imread('lenna.png', 0)
img_noise = SaPep_Noise(img, 0.2)

img1 = cv.imread('lenna.png')
img2 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_noise = SaPep_Noise(img2, 0.5)
#显示图像
img = cv.imread('lenna.png', 0)
cv.imshow('image', img) #不加上面一句，也出现椒盐
cv.imshow('img_noise', img_noise)
cv.imshow('img2_noise', img2_noise)
cv.waitKey(0)
cv.destroyWindow('all')
