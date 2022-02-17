import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

result3 = img.copy()


'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [310, 0], [0, 580], [310, 580]])
print(img.shape)
# 生成透视变换矩阵；进行透视变换
m = cv2.getPerspectiveTransform(src, dst) #返回由源图像中矩形到目标图像矩形变换的矩阵
print("warpMatrix:")
print(m)
result = cv2.warpPerspective(result3, m, (310, 580)) # 通过warpPerspective函数来进行变换
cv2.imshow("src", img)
cv2.imshow("result", result)
cv2.waitKey(0)
