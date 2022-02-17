# -*- coding: utf-8 -*-

import cv2 as cv
import numpy as np
# from numpy import shape
import random

class gaussian_WHITEBLACK():
    def gaussian1(self, src, mean, sigma, percentage):
        noiseImg = src
        noiseNum = int(percentage*src.shape[0]*src.shape[1])#报错记录1，未转int
        for i in range(noiseNum):
            randROW = random.randint(0, src.shape[0]-1)
            randCLUM = random.randint(0, src.shape[1]-1)
            noiseImg[randROW, randCLUM] = noiseImg[randROW, randCLUM] + random.gauss(mean, sigma)

            if(noiseImg[randROW, randCLUM] < 0):
                noiseImg[randROW, randCLUM] = 0
            elif(noiseImg[randROW, randCLUM] >255):
                noiseImg[randROW, randCLUM] = 255
        return noiseImg

    def setRange(self, data):
        if(data > 255):
            return 255
        elif(data < 0):
            return 0
        else:
            return data

    def gaussian2(self, src):
        img = src
        h, w, c = img.shape
        for row in range(h):
            for clum in range(w):
                gaussNum = np.random.normal(10, 20, 3)
                b = img[row, clum, 0]
                g = img[row, clum, 1]
                r = img[row, clum, 2]

                img[row, clum, 0] = gaussian_WHITEBLACK().setRange(b + gaussNum[0])
                img[row, clum, 1] = gaussian_WHITEBLACK().setRange(g + gaussNum[1])
                img[row, clum, 2] = gaussian_WHITEBLACK().setRange(r + gaussNum[2])
        return img

    def whiteBLACK(self, src, percentage):
        noiseImg = src
        noiseNum = int(percentage*src.shape[0]*src.shape[1])
        for i in range(noiseNum):
            randomX = random.randint(0, src.shape[0]-1)
            randomY = random.randint(0, src.shape[1]-1)

            if random.random()<=0.5:
                noiseImg[randomX, randomY] = 0
            else:
                noiseImg[randomX, randomY] = 255
        return noiseImg


if __name__ == '__main__':
    # img = cv.imread("lenna.png", 0)
    # img1 = gaussian_WHITEBLACK().gaussian1(img, 1, 2, 0.9)
    # cv.imshow("src", img)
    # cv.waitKey(0)
    # cv.imshow("dst", img1)
    # cv.waitKey(0)#报错记录2，waitKey的K要大写
    # cv.destroyAllWindows()

    # img = cv.imread("lenna.png")
    # cv.imshow("gaussian", gaussian_WHITEBLACK().gaussian2(img))#报错记录1，需要mat参数
    # cv.waitKey(0)

    img = cv.imread("lenna.png")
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow("src", imgray)
    cv.waitKey(0)
    cv.imshow("whiteBLACK", gaussian_WHITEBLACK().whiteBLACK(imgray, 0.8))#报错记录1，没添加percentage
    cv.waitKey(0)
    cv.destroyAllWindows()





