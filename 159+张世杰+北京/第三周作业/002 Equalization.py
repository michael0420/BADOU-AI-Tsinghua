# encoding: utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
函数的方式实现图像均衡化
"""
"""
灰色图像
"""
img3 = cv2.imread('lenna.png', 1)  # 读取彩色图像
img3_gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)  # 灰度化
img3_gray2 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)  # 灰度化

(H, W) = img3_gray.shape[:2]  # 获取图像的高和宽
for p in range(0, 256):
    ct = 0
    for i in range(H):
        for j in range(W):
            if img3_gray[i, j] <= p:
                ct += 1
    q = (ct / (H * W)) * 256 - 1
    for i in range(H):
        for j in range(W):
            if img3_gray2[i, j] == p:
                img3_gray2[i, j] = q
cv2.imshow("Histogram Equalization", np.hstack([img3_gray, img3_gray2]))
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
cv2接口的方式实现均衡化
"""
"""
灰色图像：通过equalizeHist—直方图均衡化；函数原型： equalizeHist(src, dst=None)；src：图像矩阵(单通道图像)； dst：默认即可
"""
# 导入灰色图像
img = cv2.imread('lenna.png', 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 进行直方图均衡化处理，对图片做均衡化处理；
dst = cv2.equalizeHist(img_gray)
# h绘制均衡化后的直方图
dst_hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
# 直接绘制出直方图
plt.figure()
plt.hist(dst.ravel(), 256)  # plt.plot(dst_hist)；两者展示出来的直方图形式不一样；
plt.show()
# 图像展示
cv2.imshow("Histogram Equalization", np.hstack([img_gray, dst]))
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
彩色图像：通过split函数，将三通道展开； 通过cv2.equalizeHist( )函数分别均衡化， 通过merge函数将图像进行合并；
"""
img2 = cv2.imread('lenna.png', 1)

(b, g, r) = cv2.split(img2)
bE = cv2.equalizeHist(b)
gE = cv2.equalizeHist(g)
rE = cv2.equalizeHist(r)
dst_img2 = cv2.merge((bE, gE, rE))

cv2.imshow('color_E', np.hstack([img2, dst_img2]))
cv2.waitKey(0)
cv2.destroyAllWindows()
