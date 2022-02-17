
'''
彩色图像灰度
2021/11/28--2021/11/29
'''
import cv2
import numpy as np

#读入图片
image_scr = cv2.imread("F:/cycle_gril/lenna.png")
#获取图片长和宽
hei , wid = image_scr.shape[0:2]
#创建一张空白图像，大小和读入图像一至
image_dst = np.zeros([hei,wid],image_scr.dtype)
#循环设置每个像素值
for h in range(hei):
    for w in range(wid):
        temp = image_scr[h,w] #获取一个像素点值
        image_dst[h,w] = temp[0]*0.11+temp[1]*0.59+temp[2]*0.3
cv2.imshow("image_dst",image_dst);
cv2.imshow("image_src",image_scr);

cv2.waitKey(0)


