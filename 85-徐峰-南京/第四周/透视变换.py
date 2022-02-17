import cv2 as cv
import numpy as np

img = cv.imread('D:/BaiduNetdiskDownload/photo1.jpg')

result3 = img.copy()

"""
src, 和 dst的输入并不是图像，而是图像对应的顶点坐标
"""

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
print(img.shape)

#生成透视变换矩阵，进行透视变换
m = cv.getPerspectiveTransform(src, dst)
print("变换矩阵")
print(m)

res = cv.warpPerspective(result3, m, (337, 488))
cv.imshow("src", img)
cv.imshow("res", res)
cv.waitKey(0)