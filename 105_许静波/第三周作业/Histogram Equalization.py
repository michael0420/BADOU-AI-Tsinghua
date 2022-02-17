import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png", 1)
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h,w = img_grey.shape
hist = cv2.calcHist([img_grey],[0],None,[256],[0,255])#这里返回的是次数
hist[0:255] = hist[0:255]/(h*w)
sum_hist = np.zeros(hist.shape)#用于存放灰度级别概率的累和
for i in range(256):
    sum_hist[i] = sum(hist[0:i+1])#将前i+1个灰度级别的出现概率总和赋值给sum_hist[i]

#创建映射关系
equal_hist = np.zeros(sum_hist.shape)
for j in range(256):
    equal_hist[j] = int(256*sum_hist[j]-1)

equal_img = img.copy()#用于存放均衡化后图像的灰度值
for i in range(h):
    for j in range(w):
        equal_img[i,j] = equal_hist[img_grey[i,j]]

plt.figure()
plt.hist(img_grey.ravel(), 256)
plt.hist(equal_img.ravel(), 256)
plt.show()