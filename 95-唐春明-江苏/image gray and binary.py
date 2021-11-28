import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray
from PIL import Image

#导入图片
img = cv2.imread('lenna.png')

#灰度图  -浮点算法
h,w = img.shape[:2]    #获取图片的height和width
img_gray = np.zeros([h,w],img.dtype)    #创建一张和当前图片大小一样的单通道图片
for i in range(h):
    for j in range(w):
        k = img[i,j]     #取出当前height和width下的BGR坐标
        img_gray[i,j] = int(k[0]*0.11+k[1]*0.59+k[2]*0.3)  #将BGR坐标转换成gray坐标
print(img_gray)
print("image gray: %s"%img_gray)
cv2.imshow("image gray",img_gray)
#cv2.waitKey(0)   #进程阻塞，值为0是一直显示这一帧

#灰度图  -函数实现
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna---")
print(img)
img_gray = rgb2gray(img)
img_gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray---")
print(img_gray)


#二值图  -浮点算法
rows,cols = img_gray.shape
for i in range(rows):
    for j in range(cols):
        if (img_gray[i,j]) <= 0.5:
            img_gray[i,j] = 0
        else:
            img_gray[i,j] = 1
print("---image binary---")
print(img_gray)

#img_binary = np.where(img_gray <=0.5, 0, 1)  简单写法
#print(img_binary)
#print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_gray, cmap='gray')
plt.show()


