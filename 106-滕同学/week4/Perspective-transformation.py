import cv2
import numpy as np

img = cv2.imread('chepai.jpeg')

result3 = img.copy()

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
# src = np.float32([[6, 239], [618, 125], [64, 498], [628, 293]])
src = np.float32([[6, 239], [618, 125], [628, 293], [64, 498]])
dst = np.float32([[0, 0], [600, 0], [600, 250], [0, 250]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst)
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (600, 250))
cv2.imwrite("result.png", result)