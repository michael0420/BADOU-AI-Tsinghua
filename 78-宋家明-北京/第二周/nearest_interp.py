import cv2
import numpy as np
import argparse

def fun(img, nw, nh):
    h, w, _ = img.shape
    emptyimg = np.zeros((nh,nw,3),np.uint8)
    sh, sw = nh/h, nw/w

    for jy in range(nh):
        for ix in range(nw):
            x, y = int(ix/sw), int(jy/sh)
            emptyimg[jy,ix] = img[y,x]

    return emptyimg


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='nearest function')
    parser.add_argument('--w', type=int, default=800, help='width')
    parser.add_argument('--h', type=int, default=800, help='height')
    opt = parser.parse_args()

    nw, nh = opt.w, opt.h
    img = cv2.imread('lenna.png')

    zoom = fun(img, nw, nh)

    cv2.imshow('oriimg', img)
    cv2.imshow('zoom', zoom)
    cv2.waitKey(0)
        


