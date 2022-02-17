import cv2
import numpy as np
import random
import copy
def GaussianNoise(img,means,sigma,percetage):
    img_noise = copy.deepcopy(img)
    h,w,channels = img.shape
    nums_noise = int(h*w*percetage)
    for i in range(nums_noise):
        rand_x = random.randint(0,h-1)
        rand_y = random.randint(0,w-1)
        img_noise[rand_x,rand_y] = img[rand_x,rand_y,:]+random.gauss(means,sigma)
        for j in range(3):
            if img_noise[rand_x,rand_y,j] < 0:
                img_noise[rand_x, rand_y, j] = 0
            elif img_noise[rand_x,rand_y,j] > 255:
                img_noise[rand_x, rand_y, j] = 255
    return img_noise

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img_noise = GaussianNoise(img, 2, 4, 0.8)
    cv2.imshow('source', img)
    cv2.imshow('GaussianNoise',img_noise)
    cv2.waitKey(0)