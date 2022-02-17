#coding=utf-8

import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

"""
  图像直方图均衡化
"""
def hist_eqalization(img,c):
	h,w=img.shape[:2]
	hist_img=np.zeros((h,w),dtype=np.uint8)
	hist=np.zeros((256,1),dtype=np.uint32)
	for i in range(h):
		for j in range(w):
			k=img[i,j]
			hist[k]=hist[k]+1
	# plt.figure(c)
	# plt.title("histogram")
	# plt.xlabel("Bins")
	# plt.ylabel("nums")
	# plt.plot(hist,color='r')
	# plt.xlim([0,256])
	# plt.show()
	histcum=np.zeros((256,1),dtype=np.float32)
	for k in range(256):
		histcum[k]=float(hist[k]/(h*w))
	for k in range(1,256):
		histcum[k]=histcum[k] +histcum[k-1]
	hist_result=np.zeros((256,1),dtype=np.uint32)
	for k in range(1,256):
		hist_result[k]=histcum[k]*256-1
		
	for i in range(h):
		for j in range(w):
			k=img[i,j]
			hist_img[i,j]=np.uint8(hist_result[k])
	return hist_img
			






img=cv2.imread("lenna.png",cv2.IMREAD_UNCHANGED)

(b,g,r)=cv2.split(img)
hist_eqaimgb=hist_eqalization(b,1)
hist_eqaimgg=hist_eqalization(g,2)
hist_eqaimgr=hist_eqalization(r,3)
# cv2.imshow('b',hist_eqaimgb)
# cv2.imshow('g',hist_eqaimgg)
# cv2.imshow('r',hist_eqaimgr)
# cv2.waitKey(0)
hist_eqaimg=cv2.merge((hist_eqaimgb,hist_eqaimgg,hist_eqaimgr))
cv2.imshow("hist_eqalize_img",hist_eqaimg)
cv2.waitKey(0)