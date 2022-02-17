
# chen x.z. 2021/12/09

from skimage import io
import cv2
import numpy as np

src = cv2.imread("AWACS.jpeg")
gray = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
bound = cv2.Canny(gray, 150, 280)

cv2.imshow("canny边缘检测", bound)
cv2.waitKey(5000)
cv2.destroyAllWindows()
