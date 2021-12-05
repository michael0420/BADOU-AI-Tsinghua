#最邻近插值实现

import cv2
import numpy as np

img = cv2.imread("lenna.png")   #读取图片

img_h, img_w, img_c = img.shape
img_target = np.zeros([1000, 1000, img_c], img.dtype)

prop_h = 1000 / img_h   #比例系数
prop_w = 1000 / img_w

for i in range(1000):
    for j in range(1000):
        scr_x = round(i / prop_h)
        scr_y = round(j / prop_w)
        print("i", scr_x)
        img_target[i, j] = img[scr_x, scr_y]

cv2.imshow("target", img_target)
cv2.imshow("original", img)
cv2.waitKey()





