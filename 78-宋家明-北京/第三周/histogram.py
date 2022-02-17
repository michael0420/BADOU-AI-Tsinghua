import glob
import cv2
import argparse
import numpy as np
import copy

if __name__=='__main__':
    
    img = cv2.imread('../lenna.png')
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    gray_img = r*0.3 +g*0.59 +b*0.11
    gray_img = gray_img.astype(np.uint8)

    print(gray_img.shape)
    h, w = gray_img.shape
    img_pix = {}
    for i in range(256):
        img_pix[str(i)] = 0
    for y in range(h):
        for x in range(w):
            img_pix[str(gray_img[y,x])] += 1
    print(img_pix)

    imgs_dian = h*w 
    for i in range(256):
        img_pix[str(i)] = round(img_pix[str(i)]/imgs_dian,2)
    pi = 0
    for i in range(256):
        pi += img_pix[str(i)] 
        img_pix[str(i)] = pi
    img_value = {}

    for i in range(256):
        img_value[str(i)] = round(img_pix[str(i)]*256-1)

    hisimg = np.zeros((h,w),dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            hisimg[y,x] = img_value[str(gray_img[y,x])]


    cv2.imshow('hisimg',hisimg)
    cv2.imshow('grayimg',gray_img)
    cv2.waitKey(0)
