import cv2
from matplotlib import pyplot as plt
import numpy as np


img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
dst = cv2.equalizeHist(gray)
hist1 = cv2.calcHist([gray],[0],None,[256],[0,256])
hist2 = cv2.calcHist([dst],[0],None,[256],[0,256])

# cv2.imshow("histogram equalization", np.hstack([gray,dst]))
# cv2.waitKey()

plt.figure()
plt.subplot(121)
plt.hist(gray.ravel(),256)
plt.subplot(122)
plt.hist(dst.ravel(),256)
plt.show()
