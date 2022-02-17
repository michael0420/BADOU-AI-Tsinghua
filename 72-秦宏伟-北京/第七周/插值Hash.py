#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
"""
插值Hash
"""
def difference_hash(img):
    hash = ''
    for i in range(img.shape[0]):
        for j in range(img.shape[1]-1):
            if img[i,j] > img[i,j+1]:
                hash +='1'
            else:
                hash += '0'
    print(hash)
    return hash

def hanmingDistance(str1,str2):
    #计算字符串的汉明距离
    dis = 0
    if len(str1) != len(str2):
        return -1
    for i in range (len(str1)):
        if str1[i]!=str2[i]:
           dis +=1
    print(dis)
    return dis

img = cv2.imread('lenna.png',0)
img_noise = cv2.imread('lenna_noise.png',0)


#缩小图像到8*8
img_zoom = cv2.resize(img,(9,8))
img_noise_zoom = cv2.resize(img_noise,(9,8))

img_hash = difference_hash(img_zoom)
img_noise_hash = difference_hash(img_noise_zoom)

hanmingDistance(img_hash,img_noise_hash)