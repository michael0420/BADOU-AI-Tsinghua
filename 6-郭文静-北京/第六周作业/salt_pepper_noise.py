# coding=utf-8

import numpy as np
import cv2
import random

def SaltPepperNoise(src,percetage):
    noise_img=src
    h,w=src.shape[:2]
    noise_num=int(percetage*w*h)
    for i in range(noise_num):
        #每次取一个随机点(randX,randY)
        randX = random.randint(0, w-1)
        randY = random.randint(0, h-1)
        #random.random()取值为0-1.0
        tmp = random.random()
        if tmp<=0.5:
            noise_img[randY,randX,:] = 0
        else:
            noise_img[randY,randX,:] = 255
        
    return noise_img
    
    
    
    
if __name__=='__main__':
    img=cv2.imread('lenna.png',cv2.IMREAD_ANYCOLOR)
    cv2.imshow('src', img)
    noise_img=SaltPepperNoise(img,0.8)
    cv2.imshow('SaltPepperNoise',noise_img)
    cv2.waitKey(0)