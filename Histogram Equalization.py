import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("lenna.png")
# single tunnel
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# histogram equalization
dst = cv2.equalizeHist(gray)
hist = cv2.calcHist([dst], [0], None, [256], [0, 255])
plt.figure()
plt.hist(dst.ravel(), 255)
plt.show()
# show image with gray and hist-equalized img
# cv2.imshow("histogram equalization", np.hstack([gray, dst]))
# cv2.waitKey(0)

# 3 tunnel, get data for each tunnel, then do hist equal for each tunnel
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# combine all 3 hist-equalized tunnels altogether
result = cv2.merge((bH, gH, rH))
cv2.imshow("histogram equalization", np.hstack([img, result]))
cv2.waitKey(0)