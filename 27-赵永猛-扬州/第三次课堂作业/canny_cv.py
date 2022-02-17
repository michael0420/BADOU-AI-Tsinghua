import numpy as np
import cv2

img = cv2.imread('lenna.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('img')


def nothing(x):
    pass


cv2.createTrackbar('intensity', 'img', 0, 255, nothing)
while(1):
    # 返回滑块所在位置对应的值
    intensity = cv2.getTrackbarPos('intensity', 'img')
    edge = cv2.Canny(gray_img, 100, intensity)
    cv2.imshow('img', edge)
    if cv2.waitKey(1) == ord('q'):
        break
cv2.destroyAllWindows()
