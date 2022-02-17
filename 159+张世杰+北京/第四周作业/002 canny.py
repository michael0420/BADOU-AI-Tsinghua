# encoding: utf-8

import cv2

'''
直接调用接口：v2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])
参数说明：
第一个参数：scr：必须为单通道灰度图；
第二个参数：阈值1；
第二个参数：阈值2；后面参数不详细；
'''

img = cv2.imread('lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 可省略，直接读入0 灰度图像
dst = cv2.Canny(gray, 10, 300)

cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
