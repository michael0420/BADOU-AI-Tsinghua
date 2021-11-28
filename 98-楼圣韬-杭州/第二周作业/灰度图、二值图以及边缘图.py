from typing import Any
import imutils
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image
import cv2


img = cv2.imread("lenna.png")

# 手动转灰度，利用权重：R:0.3,G:0.59,B:0.11
'''           
h,w = img.shape[:2]    # 用h，w获取元组的前两个数据
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m=img[i,j]
        img_gray[i,j]= int(m[0]*0.11+m[1]*0.59+m[2]*0.3)  #opencv读的是BGR而非RGB
cv2.imshow("image show gray", img_gray)
'''

img_gray=rgb2gray(img)
mpl.rcParams['font.sans-serif'] = 'KaiTi'
fig = plt.figure()
plt.suptitle("原图、灰度图与二值图")

plt.subplot(221)
plt.title("原图")
plt.imshow(imutils.opencv2matplotlib(img))  # 显示彩色图

plt.subplot(222)
plt.title("灰度图")
plt.imshow(img_gray,cmap='gray')   # 不加cmap='gray'输出的不是灰度图

# 手动灰度转二值

'''
rows,cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if img_gray[i,j] <= 0.5:
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1
'''

plt.subplot(223)
plt.title("二值图")
img_binary=np.where(img_gray >= 0.5,1,0)
plt.imshow(img_binary,cmap='gray')

def get_contour(bin_img):
    # get contour
    contour_img = np.zeros(shape=(bin_img.shape),dtype=np.uint8)
    contour_img += 255
    h = bin_img.shape[0]
    w = bin_img.shape[1]
    for i in range(1,h-1):
        for j in range(1,w-1):
            if(bin_img[i][j]==0):
                contour_img[i][j] = 0
                sum = 0
                sum += bin_img[i - 1][j + 1]
                sum += bin_img[i][j + 1]
                sum += bin_img[i + 1][j + 1]
                sum += bin_img[i - 1][j]
                sum += bin_img[i + 1][j]
                sum += bin_img[i - 1][j - 1]
                sum += bin_img[i][j - 1]
                sum += bin_img[i + 1][j - 1]
                if sum ==  0:
                    contour_img[i][j] = 255

    return contour_img

contour_img = get_contour(img_binary)
plt.subplot(224)
plt.title("边缘图")
plt.imshow(contour_img,cmap='gray')
plt.show()