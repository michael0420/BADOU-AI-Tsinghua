# -*- coding: utf-8 -*-
'''
@Time    : 2021/11/21/00021 21:03
@Author  : Chen Quan
@FileName: T1.py
@Software: PyCharm
'''
'''
1.灰度图、二值图实现
'''
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
img=cv2.imread("lenna.png")
img_origin=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.imshow(img_origin)
print("---image origin----")
print(img_origin)
print(img_origin.shape)


img_gray=cv2.cvtColor(img_origin,cv2.COLOR_RGB2GRAY)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
print("---image gray----")
print(img_gray)
print(img_gray.shape)

img_binary=np.where(img_gray>=128,1,0)
print("-----------image_binary----------")
print(img_binary)
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()