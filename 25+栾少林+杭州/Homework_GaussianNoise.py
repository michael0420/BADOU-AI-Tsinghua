import random

import cv2
import cv2 as cv
import numpy as np

def GaussianNoise(src,means,sigma,percentage):
    deal_img=src
    NoiseimgNum=int(percentage*deal_img.shape[0]*deal_img.shape[1])
    for i in range(NoiseimgNum):
        # 以下为随机获取图像像素，边缘点不取
        # random.randint(a,b)为在范围内随机获取一个整数，a,b分别代表取值范围上下限
        randomX=random.randint(1,deal_img.shape[0]-1)
        randomY=random.randint(1,deal_img.shape[1]-1)
        #对于当前的随机像素，增加一个符合高斯分布的噪声
        #random.gauss(mu,sigma)为获得一个符合高斯分布的随机值，mu，sigma代表高斯分布参数（均值与方差）
        deal_img[randomX,randomY]=deal_img[randomX,randomY]+random.gauss(means,sigma)
        #对于新的像素，像素值要在[0,255]范围内
        if deal_img[randomX,randomY]>255:
            deal_img[randomX,randomY]=255
        elif deal_img[randomX,randomY]<0:
            deal_img[randomX,randomY]=0
    return deal_img

test_img=cv2.imread("lenna.png")
test_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Source Img",test_img)
means=2
sigma=4
percentage=0.8
New_img=GaussianNoise(test_img,means,sigma,percentage)
cv2.imshow("GaussianNoise",New_img)
cv2.waitKey(10000)


