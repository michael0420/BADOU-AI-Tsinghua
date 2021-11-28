# -*- coding: utf-8 -*-
"""
@author:gwj

彩色图像转灰度，并做二值化
"""

import numpy as np
import cv2

# rgb2gray
img = cv2.imread('lenna.png')   #load image
h,w ,c= img.shape
gray_lena=np.zeros([h,w],img.dtype)#create image
for i in range(h):
	for j in range(w):
		gray_lena[i,j]=int(img[i,j,0]*0.11+img[i,j,1]*0.59+img[i,j,2]*0.3)# bgr2gray

cv2.imshow("gray_lena",gray_lena.astype("uint8"))
cv2.waitKey(0)

#binary lena

binary_lena=np.zeros([h,w],gray_lena.dtype)
thresh=100
for i in range(h):
	for j in range(w):
		if(gray_lena[i,j]>thresh): #global thresh
			binary_lena[i,j]=255
		else:
			binary_lena[i,j]=0




cv2.imshow("binary_lena",binary_lena.astype("uint8"))
cv2.waitKey(0)


