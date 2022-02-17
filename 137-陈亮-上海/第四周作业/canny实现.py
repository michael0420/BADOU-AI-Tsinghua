import cv2
import numpy as np

# 读图
img = cv2.imread('lenna.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# 高斯平滑
gauss_img = cv2.GaussianBlur(img_gray,(3,3),0)
# canny检测边缘
img_canny = cv2.Canny(gauss_img,100,200)
# 显示
cv2.imshow('img_canny',img_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
