# -*- coding: utf-8 -*-
'''
@Time    : 2021/11/21/00021 21:12
@Author  : Chen Quan
@FileName: T2.py
@Software: PyCharm
'''
'''
2.最邻近插值实现
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
def transform(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def function(img,h,w):
    height,width,channels=img.shape
    emptyImage=np.zeros((h,w,channels),np.uint8)
    sh=h/height
    sw=w/width
    for i in range(h):
        for j in range(w):
            x=int(i/sh)
            y=int(j/sw)
            emptyImage[i,j]=img[x,y]

    return emptyImage

img=cv2.imread("lenna.png")
img_origin=transform(img)
print(img_origin)
print(img_origin.shape)
plt.subplot(121)
plt.imshow(img_origin)
zoom=function(img_origin,1000,1000)
plt.subplot(122)
plt.imshow(zoom)
plt.show()
print(zoom)
print(zoom.shape)



