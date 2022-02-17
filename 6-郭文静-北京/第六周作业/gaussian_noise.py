# coding=utf-8

import numpy as np
import cv2
import random

def GaussianNoise(src, means, sigma, percetage):
    noise_img = src
    h,w = src.shape[:2]
    noise_num=int(percetage * w * h)
    for i in range(noise_num):
    #每次取一个随机点(randX,randY)
        randX = random.randint(0, w-1)
        randY = random.randint(0, h-1)
        tmp = noise_img[randY,randX,:]+random.gauss(means, sigma)
        if tmp.any()<0:
             noise_img[randY, randX, :] = 0
        elif tmp.any()>255:
            noise_img[randY, randX, :] = 255
        else:
            noise_img[randY, randX, :] = tmp
    return noise_img

if __name__=='__main__'	:
    img=cv2.imread('lenna.png',cv2.IMREAD_ANYCOLOR)
    cv2.imshow('src', img)
    noise_img=GaussianNoise(img,2,4,0.8)

    cv2.imshow('gaussian_noise',noise_img)
    cv2.waitKey(0)