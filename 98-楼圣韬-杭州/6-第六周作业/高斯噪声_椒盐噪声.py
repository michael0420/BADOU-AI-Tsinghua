import cv2
import numpy as np
import random


def nothing(x):
    pass

# --------------------高斯噪声-----------------------#
def Gsnoise(img,means, sigma, per):
    p = img.copy()
    Noisenum = int(per * img.shape[0] * img.shape[1])  # 噪声比
    for i in range(Noisenum):
        x = random.randint(1, img.shape[0] - 1)  # 高斯噪声不处理边缘，所以减1
        y = random.randint(1, img.shape[1] - 1)  # 取一个随机点
        # 在此处原有的像素灰度值上加上随机数
        p[x, y] = img[x, y] + random.gauss(means, sigma)
        # random.gauss(means,sigma) means为平均数，sigma为标准偏差，返回的是高斯分布浮点数
        if p[x, y] < 0:
            p[x, y] = 0
        elif p[x, y] > 255:
            p[x, y] = 255
    return p

def callback(x):
    pass

img = cv2.imread('lenna.png', 0)
cv2.namedWindow('GaussianNoise')
cv2.createTrackbar('percentage%', 'GaussianNoise', 80, 99, callback)
cv2.createTrackbar('means', 'GaussianNoise', 20, 200, callback)
cv2.createTrackbar('sigma', 'GaussianNoise', 10, 100, callback)

while 1:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    per=cv2.getTrackbarPos('percentage%', 'GaussianNoise')/100
    means=cv2.getTrackbarPos('means', 'GaussianNoise')
    sigma=cv2.getTrackbarPos('sigma', 'GaussianNoise')
    cv2.imshow('GaussianNoise', Gsnoise(img,means,sigma,per))
cv2.destroyAllWindows()
# --------------------高斯噪声-----------------------#


# --------------------椒盐噪声-----------------------#
def spnoise(img,per1):
    p=img.copy()
    sum=int(per1*img.shape[0]*img.shape[1])
    for i in range(sum):
        x=random.randint(1,img.shape[0]-1)
        y=random.randint(1,img.shape[1]-1)
        if random.random()<=0.5:    # random()生成0-1之间的随机浮点数
            p[x,y]=0
        else:
            p[x,y]=255
    return p

cv2.namedWindow('spnoise')
cv2.createTrackbar('percentage%', 'spnoise', 30, 99, callback)

while 1:
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    per=cv2.getTrackbarPos('percentage%', 'spnoise')/100
    cv2.imshow('spnoise', spnoise(img,per))
cv2.destroyAllWindows()