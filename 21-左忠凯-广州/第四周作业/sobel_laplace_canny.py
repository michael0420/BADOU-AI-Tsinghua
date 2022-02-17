import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图片
img = cv2.imread("lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 使用sobel算子求边缘
img_sobel_x = cv2.convertScaleAbs(cv2.Sobel(img_gray, cv2.CV_16S, 1, 0))
img_sobel_y = cv2.convertScaleAbs(cv2.Sobel(img_gray, cv2.CV_16S, 0, 1))
img_sobel = cv2.addWeighted(img_sobel_x, 0.5, img_sobel_y, 0.5, 0)

# 使用laplace算子计算,
'''
    使用Laplacian函数计算，函数原型：
    Laplacian(src, ddepth, dst, ksize, scale, delta, borderType)，函数参数如下：
    src：原始图像
    ddepth：数据格式
    dst：目标图像
    ksize：laplace算子大小
    scale：缩放比例
    delta：可选增量
    borderType：判断图像边界的模式，默认为BORDER_DEFAULT
'''
img_laplace = cv2.convertScaleAbs(cv2.Laplacian(img_gray, cv2.CV_16S, ksize=3))

# 使用Canny算子计算
img_canny = cv2.convertScaleAbs(cv2.Canny(img_gray, 100, 150))

plt.figure(figsize=(6, 8), dpi=100)        # 画布10*10寸，dpi=100
plt.subplots_adjust(wspace=0.3, hspace=0.3) # 子图横竖间隔0.3英寸

# 显示原始灰度图
plt.subplot(3, 2, 1)
plt.imshow(img_gray, cmap='gray')
plt.title("Origin gray img")

# 显示sobel X轴
plt.subplot(3, 2, 2)
plt.imshow(img_sobel_x, cmap='gray')
plt.title("sobel X")

# 显示sobel Y轴
plt.subplot(3, 2, 3)
plt.imshow(img_sobel_y, cmap='gray')
plt.title("sobel Y")

# 显示sobel
plt.subplot(3, 2, 4)
plt.imshow(img_sobel, cmap='gray')
plt.title("sobel")

# 显示laplace
plt.subplot(3, 2, 5)
plt.imshow(img_laplace, cmap='gray')
plt.title("laplace")

# 显示canny
plt.subplot(3, 2, 6)
plt.imshow(img_canny, cmap='gray')
plt.title("canny")

plt.show()
