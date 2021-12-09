#透视变换

import cv2
import numpy as np


img = cv2.imread("photo1.jpg",1)
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])    #顶点寻找
dst = np.float32([[0, 0], [555, 0], [0, 888], [555, 888]])

#生成透视矩阵
matrix = cv2.getPerspectiveTransform(src, dst)
#生成矫正图像
img_result = cv2.warpPerspective(img, matrix, [555, 888])
cv2.imshow("img", img)
cv2.imshow("result", img_result)
cv2.waitKey()
