'''
高斯噪声
'''

import numpy
import cv2
from numpy import shape
import random

def GaussNoise(src_img,sigma,means,percentage):
    Noise_img = src_img
    Noise_number = int(percentage *Noise_img.shape[0] * Noise_img.shape[1])
    for i in range(Noise_number):
        #随机生成像素点坐标，random.randint()生成随机整数
        x = random.randint(0,Noise_img.shape[0]-1)
        y = random.randint(0,Noise_img.shape[1]-1)

        #加高斯噪声
        Noise_img[x,y] = Noise_img[x,y] + random.gauss(means,sigma) #产生高斯随机分布的噪声
        #修正超出取值范围的像素  <0 或  >250
        if Noise_img[x,y]<0:
            Noise_img[x,y]=0
        elif Noise_img[x,y]>255:
            Noise_img[x, y] = 255
    return Noise_img

src_img = cv2.imread('F:/cycle_gril/lenna.png',0)
dst_img = GaussNoise(src_img,4,2,0.7)

src_img1 = cv2.imread('F:/cycle_gril/lenna.png')
dst_img1 = cv2.cvtColor(src_img1,cv2.COLOR_BGR2GRAY)

cv2.imshow("src_img",dst_img1)
cv2.imshow("dst_img",dst_img)



cv2.waitKey(0)