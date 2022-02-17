import cv2
import numpy as np

img = cv2.imread('photo1.jpg')

# resulttmp = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resulttmp = gray


src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
# dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])
dst = np.float32([[0, 0], [300, 0], [0, 458], [300, 458]])
print(img.shape)
m = cv2.getPerspectiveTransform(src, dst)
# print("warpMatrix:")
# print(m)
result = cv2.warpPerspective(resulttmp, m, (337, 488))
cv2.imshow("src", gray)
cv2.imshow("result", result)
cv2.waitKey(0)
