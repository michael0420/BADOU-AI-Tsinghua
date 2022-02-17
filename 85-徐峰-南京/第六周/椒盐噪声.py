import numpy as np
import cv2
import random
from numpy import shape


def jiaoyan(src, percentage):
    NoiseImg = src

    #噪声点个数
    NoiseNum = int(percentage*src.shape[0]*src.shape[1])

    #每次取一个随机点
    #把一张图片的像素用行和列表示的话，randX 代表随机生成的行，randY代表随机生成的列
    #random.randint生成随机整数
    #椒盐噪声图片边缘不处理，故-1
    randX = random.randint(0, src.shape[0] - 1)
    randY = random.randint(0, src.shape[1] - 1)
    for i in range(NoiseNum):
        if random.random() <= 0.5:
            NoiseImg[randX, randY] = 0
        else:
            NoiseImg[randX, randY] = 255
    return NoiseImg


img = cv2.imread('../../../../../BaiduNetdiskDownload/lenna.png', 0)
img1 = jiaoyan(img, 0.2)

img = cv2.imread('../../../../../BaiduNetdiskDownload/lenna.png')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("source", img2)
cv2.imshow('len_noise', img1)
cv2.waitKey(0)









