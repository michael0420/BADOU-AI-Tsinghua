import cv2
import numpy as np

def hist(img_gray):

    '''计算像素对应的直方图'''
    rows,cols = img_gray.shape
    prob = np.zeros(shape=(256))
    for rv in img_gray:
        for cv in rv:
            prob[cv] += 1
    prob = prob / ( rows*cols )

    #求累加概率
    prob = np.cumsum(prob)
    # 公式：q = sigma(input_K/H*W * 256 -1)
    new_img = [int(256*prob[i]-1) for i in range(256)]
    #像素值映射
    for row in range(rows):
        for col in range(cols):
            img_gray[row,col] = new_img[img_gray[row,col]]

    return img_gray

img_gray = cv2.imread('lenna.png',0)
img = hist(img_gray)
cv2.imshow('img',img)
cv2.waitKey(0)
