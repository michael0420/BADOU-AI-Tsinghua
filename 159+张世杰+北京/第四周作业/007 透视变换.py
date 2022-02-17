# encoding: utf-8
import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
result3 = img.copy()

'''输入图像'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)

'''调用接口：获得透视变换矩阵'''
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)

'''调用接口：转换图像'''
result = cv2.warpPerspective(result3, m, (337, 488))

'''输出图像'''
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
