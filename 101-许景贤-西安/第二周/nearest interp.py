# -*- coding: utf-8 -*-
"""
@author: 101xjx

最近邻插值。w h是目标图大小。img是要插值的原图，运行时可直接修改名称即可。
"""
import cv2
import numpy as np
def function(img,w,h):
    height,width,channels =img.shape
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
h = 700;
w = 700;
zoom=function(img,w,h)
# print(zoom)# 输出矩阵
# print(zoom.shape)# (800,800,3)  高，宽，通道数
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)

