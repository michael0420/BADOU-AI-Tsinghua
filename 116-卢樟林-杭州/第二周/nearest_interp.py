#!/usr/bin/python
# -*- encoding: utf-8 -*-
'''
Created on 2021/12/11 21:42:11
@Author : LuZhanglin
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt


def nearest(img):
    h, w, channels = img.shape
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    sh = 800 / h
    sw = 800 / w
    for i in range(800):
        for j in range(800):
            x = int(i/sh)
            y = int(j/sw)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("lenna.png")
zoom = nearest(img)
print(zoom)
print(zoom.shape)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(zoom, label="nearest interp")
ax[1].imshow(img, label="image")
plt.show()