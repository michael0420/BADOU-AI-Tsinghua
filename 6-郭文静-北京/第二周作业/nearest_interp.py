# -*- coding: utf-8 -*-
"""
@author:gwj

最近邻差值缩放图像
"""

import numpy as np
import cv2

def near_interp(src,dst):
	h,w,c=src.shape
	dsth,dstw,dstc=dst.shape
	scalew=w/dstw
	scaleh=h/dsth
	for i in range(dsth):
		for j in range(dstw):
			srci = int(i*scaleh)
			srcj = int(j*scalew)
			dst[i,j]=src[srci,srcj]
		   
img=cv2.imread("lenna.png")
dsth = 700
dstw = 700
dst=np.zeros((dsth,dstw,3),img.dtype)
near_interp(img,dst)
cv2.imshow("src",img)
cv2.imshow("dst",dst)
cv2.waitKey(0)
	 
