import cv2
import copy
import numpy as np
import random

def fun(img,SNR):
    h,w,channels = img.shape
    img_noise = copy.deepcopy(img)
    nums_noise = int(h*w*SNR)
    for i in range(nums_noise):
        rand_x = random.randint(0,h-1)
        rand_y = random.randint(0,w-1)
        if random.random() < 0.5:
            img_noise[rand_x,rand_y] = [0,0,0]
        else:
            img_noise[rand_x, rand_y] = [255,255,255]
    return img_noise

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_noise = fun(img,0.8)
    cv2.imshow('source',img)
    cv2.imshow('noise', img_noise)
    cv2.waitKey(0)