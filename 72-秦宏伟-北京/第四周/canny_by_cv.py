#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基于cv实现Canny"""
import cv2


def main(fileName):
    # 1、转为灰度图
    img = cv2.imread(fileName, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('res')
    cv2.createTrackbar('min', 'res', 0, 25, nothing)
    cv2.createTrackbar('max', 'res', 0, 25, nothing)
    while (1):
        if cv2.waitKey(1) & 0xFF == 27:
            break
        maxVal = cv2.getTrackbarPos('max', 'res')
        minVal = cv2.getTrackbarPos('min', 'res')
        canny = cv2.Canny(gray, 10 * minVal, 10 * maxVal)
        cv2.imshow('res', canny)
    cv2.destroyAllWindows()

def nothing(x):
    pass

if __name__ == '__main__':
    main("lenna.png")