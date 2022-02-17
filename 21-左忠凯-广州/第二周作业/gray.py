'''图形灰度化、二值化'''

# skimage库为python的数字图像处理库
# numpy库为phython的数组与矩阵运算库
# matplotlib库用于图形绘制，如表格，正弦余弦图形
# PIL库是图形处理库

import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# 灰度化
def my_img2gray(img_rgb):
    height = img_rgb.shape[0]           # 图像长度
    width = img_rgb.shape[1]            # 图形宽度
    print("height = {}, width = {}".format(height, width))
    img_gray = np.zeros([height, width], img.dtype)  # 创建一个与原图像大小相同的0数组

    # 将原始图像灰度化
    for i in range(height):
        for j in range(width):
            pixel = img_rgb[i, j]
            img_gray[i, j] = pixel[0]*0.3 + pixel[1]*0.59 + pixel[2]*0.11 # 灰度化
    return img_gray

img = cv2.imread("lenna.png")   # 读取图像
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 转为RGB图像
img_gray = my_img2gray(img_rgb)
print(img_gray)
cv2.imshow("image show gray", img_gray)

# 二值化函数
def my_img2binary(img_gray):
    height = img_gray.shape[0]  # 图像长度
    width = img_gray.shape[1]  # 图形宽度
    img_binary = np.zeros([height, width], img_gray.dtype)  # 创建一个与原图像大小相同的0数组
    for i in range(height):
        for j in range(width):
            if img_gray[i, j] <= 0.5:
                img_binary[i, j] = 0
            else:
                img_binary[i, j] = 1
    return img_binary

# 绘制图形表格，将原始图像，灰度图，二值图显示到同一个画布上

# 1、显示原始图像
plt.subplot(2, 2, 1)
img = plt.imread("lenna.png")
plt.title("source img")
plt.imshow(img)
print(img)

# 2、显示灰度图像
plt.subplot(2, 2, 2)
img_gray = rgb2gray(img)
plt.title("gray img")
plt.imshow(img_gray, cmap='gray')
print(img_gray)

#3、显示二值化图像
plt.subplot(2, 2, 3)
img_binary = my_img2binary(img_gray)
#img_binary = np.where(img_gray >= 0.5,1,0)
plt.title("binary img")
plt.imshow(img_binary, cmap='gray')

plt.show()
cv2.waitKey(0)











