
"""
@damion  汤桂平
"""


import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

img_src = cv2.imread("lenna.png")
h,w = img_src.shape[0:2]  # 把原图的高和宽分别赋给h,w
img_gray = np.zeros([h,w],img_src.dtype)
for i in range(h):
    for j in range(w):
        m = img_src[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   # 不转换成整型也可以正常显示
print(img_src)
print(img_gray)
cv2.imshow("image show gray",img_gray)  # 若没有后面两行代码，图片将无法正常显示
cv2.waitKey(0)
cv2.destroyAllWindows()

# 使用skimage和matplotlib函数
plt.subplot(221)
img = plt.imread("lenna.png")  # 用plt.imread读取图像，元素值在0-1之间
plt.imshow(img)
print("show source image:\n")
print(img)
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')  # 没有cmap参数就会显示陈RGB空间图像
print("show gray image:\n")
print(img)

# row, col = img.shape[:2]
# for i in range(row):
#    for j in range(col):
#        if (img[i, j] >= 0.6):   这一行代码错在哪
#            img[i, j] = 1
#        else:
#            img[i, j] = 0

img_binary = np.where(img_gray >= 0.4, 1, 0)
plt.subplot(223)
print("show binary image:\n", img_binary)
plt.imshow(img_binary, cmap='gray')
plt.show()