# encoding: utf-8
import cv2
import numpy as np

img = cv2.imread("4.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
'''
nfeatures =0
nOctaveLayers =3
contrastThreshold = 0.04
edgeThreshold = 10
sigma =1.6
'''
sift = cv2.xfeatures2d.SIFT_create()
keypoints, descriptor = sift.detectAndCompute(gray, None)  # 找出关键点和进行关键点描述
print(descriptor)
print(descriptor.shape)

# 在原图上绘制初关键点
img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
                        color=(51, 163, 236))

cv2.imshow('sift_keypoints', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
