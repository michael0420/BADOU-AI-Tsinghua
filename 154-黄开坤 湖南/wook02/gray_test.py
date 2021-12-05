#coding:utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray


"""彩色图像的灰度图、二值图"""

# 1、灰度图操作实现，浮点算法：Gray=R*0.3+G*0.59+b*0.11
image1 = cv2.imread('lenna.png')     #cv2.imread()读取图片后已多维数组的形式保存图片信息，前两维表示图片的像素坐标，最后一维表示图片的通道索引，具体图像的通道数由图片的格式来决定
height, width = image1.shape[:2]       # shape(512, 512, 3),
img_gray = np.zeros([height,width], image1.dtype)   #创建一张与iamge1一样大小的单通道灰度图
for i in range(height):
    for j in range(width):
        m = image1[i, j]            #获取当前行、列的BGR位置坐标
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)    #转换成gray并赋值给心得图像
# print(img_gray)
print('image show gray: %s'%img_gray)
cv2.imshow('image show gray:', img_gray)    #cv2.imShow()函数可以在窗口中显示图像。第一个参数是一个窗口名称（也就是我们对话框的名称），它是一个字符串类型。第二个参数是我们的图像。您可以创建任意数量的窗口，但必须使用不同的窗口名称。

plt.subplot(221)            ##表示将整个图像窗口分为2行2列,当前位置为1,第一行的左图.#表示将整个图像窗口分为2行2列,当前位置为2,第一行的右图
img = plt.imread('lenna.png')
# img = cv2.imread("lenna.png", False)
plt.imshow(img)
print("---image lenna----")
print(img)              #0~1

# plt.show()

# 2、灰度图接口调用
image2 = plt.imread('lenna.png')
img_gray = rgb2gray(image2)                           #方式一
# img_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)   #方式二
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')   #cmap是colormap的简称，用于指定渐变色，默认的值为viridis(翠绿色)
print('------iamge gray2-----')
print(img_gray)


##二值化
img1 = cv2.imread('lenna.png')
img1_gray = rgb2gray(img1)
# print(img1.shape)       #(512, 512, 3)

#方式一:
# rows, cols = img1_gray.shape[:2]
# for i in range(rows):
#     for j in range(cols):
#         if img1_gray[i,j] <= 0.5:
#             img1_gray[i,j] = 0
#         else:
#             img1_gray[i,j] = 1
# plt.subplot(223)
# plt.imshow(img1_gray, cmap='gray')

#方式二
img1_binary = np.where(img1_gray >= 0.5, 1, 0)      #np.where(condition, x, y)
print('------image binary------')
print(img1_binary, img1_binary.shape)
plt.subplot(223)
plt.imshow(img1_binary, cmap='gray')

plt.show()