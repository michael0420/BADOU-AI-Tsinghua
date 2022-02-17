import cv2
import numpy as np
import matplotlib.pyplot as plt


low_threshold = 0
max_low_threshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("canny demo")

def Canny_threshold(low_threshold):
    filter = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_deges = cv2.Canny(filter, low_threshold, low_threshold * ratio, apertureSize=kernel_size)

    dst = cv2.bitwise_and(img, img, mask=detected_deges)
    cv2.imshow("canny demo", dst)

cv2.createTrackbar("Min track", "canny demo", low_threshold, max_low_threshold, Canny_threshold)

Canny_threshold(0)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows() 