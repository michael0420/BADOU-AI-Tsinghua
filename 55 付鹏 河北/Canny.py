import cv2
import numpy as np
img = cv2.imread('F:/lenna.png', 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('canny', cv2.Canny(gray, 100, 300))
cv2.waitKey()
