#!/usr/bin/env python 
# coding:utf-8

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray

# 加载原图
img = plt.imread("lenna.png")
# 设置图为2行2列,第一个位置显示
plt.subplot(221)
# 对图像进行处理
plt.imshow(img)
# 显示处理后的图像
# plt.show()


# 灰度图
# rgb图片灰度化
img_gray = rgb2gray(img)
# 设置为第二个位置显示
plt.subplot(222)
# 处理为灰色
plt.imshow(img_gray, cmap="gray")
# plt.show()


# 二值图
# 设置为第三个位置显示
plt.subplot(223)
# 如果某项>=0.5,则赋值为1,否则,赋值为0
img_binary = np.where(img_gray >= 0.5, 1, 0)
plt.imshow(img_binary, cmap="gray")
print(img_binary)
print(img_binary.shape)
plt.show()

