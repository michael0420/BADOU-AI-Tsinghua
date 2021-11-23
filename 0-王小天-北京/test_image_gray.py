# -*- coding: utf-8 -*-
"""
@author: Michael

彩色图像的灰度化、二值化
"""
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 灰度化
img = cv2.imread("lenna.png")
#imread函数有两个参数，第一个参数是图片路径，第二个参数表示读取图片的形式，有三种：
#cv2.IMREAD_COLOR：加载彩色图片，这个是默认参数，可以直接写1。
#cv2.IMREAD_GRAYSCALE：以灰度模式加载图片，可以直接写0。
#cv2.IMREAD_UNCHANGED：包括alpha，可以直接写 - 1
#cv2.imread()读取图片后已多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引，具体图像的通道数由图片的格式来决定
#print(img)
h,w = img.shape[:2]            #获取图片的high和wide
# 图像的大小，元组（行，列，通道）   image.shape
img_gray = np.zeros([h,w],img.dtype)                   #创建一张和当前图片大小一样的单通道图片
#print(w)
for i in range(h):
    for j in range(w):
        m = img[i,j]                             #取出当前high和wide中的BGR坐标
        #print(m)
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   #将BGR坐标转化为gray坐标并赋值给新图像
#print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)


plt.subplot(221)
img = plt.imread("lenna.png") 
# img = cv2.imread("lenna.png", False) 
plt.imshow(img)
#print("---image lenna----")
#print(img)

# 灰度化
img_gray = rgb2gray(img)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_gray = img
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
#print("---image gray----")
#print(img_gray)

# 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1
 
img_binary = np.where(img_gray >= 0.5, 1, 0) 
#print("-----imge_binary------")
#print(img_binary)
#print(img_binary.shape)

plt.subplot(223) 
plt.imshow(img_binary, cmap='gray')
plt.show()
