#coding:utf8

'''
灰度图与二值图
'''

#灰度化自实现
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

img=cv2.imread('lenna.png')
# print(img.shape[:2])
# print(img.dtype)
img_gray=np.zeros(img.shape[:2],img.dtype)
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img_gray[i,j]=int(img[i,j][0]*0.11+img[i,j][1]*0.59+img[i,j][2]*0.3)
# print(img_gray)
# cv2.imshow("grayimg",img_gray)
# cv2.waitKey(100)

'''
库自带的灰度方法有:
skimage.color库的rgb2gray()
cv2.cvColor(img,cv2.Color_BGR2GRAY)
'''
#灰度自带
img_gray1=rgb2gray(img)
# cv2.imshow("gray1",img_gray1)
# cv2.waitKey(100)

img_gray2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray2",img_gray2)
# cv2.waitKey(0)

plt.subplot(321)
img=plt.imread("lenna.png")
plt.imshow(img)
# plt.show()

plt.subplot(322)
plt.imshow(img_gray,cmap='gray')
# print(img_gray.shape)

plt.subplot(323)
plt.imshow(img_gray1,cmap='gray')
# print(img_gray1.shape)

plt.subplot(324)
plt.imshow(img_gray2,cmap='gray')
# print(img_gray2.shape)

# plt.show()

'''
根据某个值将所有的像素点进行二分
可分成任意值，不是必须为0 ，1 只是一般取0,1
'''
#二值化
for row in range(img_gray.shape[0]):
    for col in range(img_gray.shape[1]):
        if img_gray[row,col]>=128:
            img_gray[row,col]=255
        else:
            img_gray[row,col]=0
print("---img_gray---")
print(img_gray)

img_binary=np.where(img_gray>=128,255,0)
img_binary1=np.where(img_gray>=128,0,255)
print("---img_binary---")
print(img_binary)

print("---img_binary1---")
print(img_binary1)

plt.subplot(325)
plt.imshow(img_binary1,cmap='gray')

plt.subplot(326)
plt.imshow(img_binary,cmap='gray')
plt.show()