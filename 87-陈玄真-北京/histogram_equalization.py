
# 图像读取、显示、保存子模块导入
from skimage import io
# 颜色空间转换子模块导入
from skimage.color import rgb2gray
# 导入numpy数值计算库
import numpy as np
# 绘图工具
import matplotlib.pyplot as plt
# ......
from PIL import Image
# 导入opencv
import cv2

# 以灰度图方式读取图像并显示一段时间
# src_gray = io.imread("AWACS.jpeg", as_gray=True)
src = cv2.imread("AWACS.jpeg")
src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
# cv2.imshow("KJ-2000", src_gray)
# io.imshow(src_gray)
# cv2.waitKey(2000)

# 直方图均衡化
dst_gray = cv2.equalizeHist(src_gray)


# 对比均衡化前后的灰度直方图
fig1 = plt.figure()
ax11 = fig1.add_subplot(1, 2, 1)
ax11.hist(src_gray.ravel(), 256)
plt.hist(src_gray.ravel(), 256)
# plt.grid(True)
ax12 = fig1.add_subplot(1, 2, 2)
hist = cv2.calcHist([dst_gray],[0],None,[256],[0,256])
ax12.hist(dst_gray.ravel(), 256)
plt.hist(dst_gray.ravel(), 256)
# plt.grid(True)
# cv2.waitKey(2000)

# 对比均衡化前后的图像
fig2 = plt.figure()
ax21 = fig2.add_subplot(1, 2, 1)
io.imshow(src_gray)
ax22 = fig2.add_subplot(1, 2, 2)
io.imshow(dst_gray)
# cv2.waitKey(2000)

plt.show()
# cv2.destroyAllWindows()

