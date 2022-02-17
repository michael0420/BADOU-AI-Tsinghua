# 随机生成符合正态（高斯）分布的随机数，means,sigma为两个参数
import numpy as np
import cv2
from numpy import shape 
import random


def GaussianNoise(src, means, sigma, percetage):
    NoiseImg = src
    NoiseNum = int(percetage * src.shape[0] * src.shape[1])
    for i in range(NoiseNum):
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        NoiseImg[randX, randY] = NoiseImg[randX, randY] + random.gauss(means, sigma)
        if  NoiseImg[randX, randY] < 0:
            NoiseImg[randX, randY] = 0
        elif NoiseImg[randX, randY] > 255:
            NoiseImg[randX, randY] = 255

    return NoiseImg
    

img = cv2.imread('lenna.png', 0)
img_gausenoise = GaussianNoise(np.copy(img), 10, 20, 1)
cv2.imwrite("src.png", img)
cv2.imwrite("GaussianNoise.png", img_gausenoise)