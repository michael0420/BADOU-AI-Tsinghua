# encoding: utf-8


from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


"""
图像灰度化处理
"""
img = cv2.imread("lenna.png")  # 读取图片
h, w = img.shape[:2]  # 获取图片的高和宽， 图像处理过程中都是先读取行（高），再读取列（宽）；
img_gray = np.zeros([h, w], img.dtype)  # 创建空白矩阵
for i in range(h):
    for j in range(w):
        m = img[i, j]
        img_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
# print(img_gray)
cv_show('img_gray1', img_gray)

# 灰度化2
img_gray2 = rgb2gray(img)
cv_show('img_gray2', img_gray2)

# 灰度化3
img_gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('img_gray3', img_gray3)

"""
二值化图片
"""
# 手写方法
rows, cols = img_gray2.shape
for i in range(rows):
    for j in range(cols):
        if img_gray2[i, j] <= 0.5:  # 这个阶段的图片格式是float64, 此处存在问题，打印出来为纯黑图像 可以print(img.dtype)
            img_gray2[i, j] = 0
        else:
            img_gray2[i, j] = 1
cv_show('bin1', img_gray2)
# 直接调用端口
img_bin = img_binary = np.where(img_gray2 >= 0.5, 1, 0)
