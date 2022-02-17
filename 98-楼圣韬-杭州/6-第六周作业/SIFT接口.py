import cv2
import numpy as np

img = cv2.imread("lenna.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)


img=cv2.drawKeypoints(gray,keypoints,img)  # gray 目标图像，keypoints 为sift求出的特征点集 画在img里

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
