import cv2
import numpy as np
from matplotlib import pyplot as plt
# img = cv2.imread('F:/lenna.png', 1)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# dst = cv2.equalizeHist(gray)
# hist = cv2.calcHist([dst], [0], None, [256], [0, 256])
# plt.figure()
# plt.hist(dst.ravel(), 256)
# plt.show()
#
# cv2.imshow('HE', np.hstack([gray, dst]))
# cv2.waitKey(0)

img = cv2.imread('F:/lenna.png', 1)
(b, g, r) = cv2.split(img)
B = cv2.equalizeHist(b)
G = cv2.equalizeHist(g)
R = cv2.equalizeHist(r)
Img = cv2.merge((B, G, R))
cv2.imshow('Img', Img)
cv2.waitKey()
