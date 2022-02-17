import random

import cv2
import numpy as np
import cv2 as cv

def PepperSalt_Noise(src,SNR):
    deal_img=src
    deal_img=cv.cvtColor(src,cv2.COLOR_BGR2GRAY)
    ImgNum=int(deal_img.shape[0]*deal_img.shape[1]*SNR)
    for i in range(ImgNum):
        randomX=random.randint(1,deal_img.shape[0]-1)
        randomY=random.randint(1,deal_img.shape[1]-1)
        if random.random()<=0.5:
            deal_img[randomX,randomY]=255
        else:
            deal_img[randomX,randomY]=0
    return deal_img

test_img=cv2.imread("lenna.png")
new_img=PepperSalt_Noise(test_img,0.5)
cv2.imshow("Source Image",test_img)
cv2.imshow("Pepper and Salt Noise",new_img)
cv2.waitKey(10000)