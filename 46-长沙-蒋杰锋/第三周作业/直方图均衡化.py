# -*- coding:utf-8 -*-
# @Time : 2021/12/1 19:44
# @Author : Sakura
# @QQEmail : 1018655370@qq.com
# @Google : jiangjiefeng0@gmail.com
import matplotlib.pyplot as plt
import numpy as np
import cv2


def HistogramEqualization(image):
    height, width = image.shape
    # image total pixel
    imageTotalPixel = height * width
    print("image total pixel is ", imageTotalPixel)

    plt.hist(image.flatten(), bins=255)
    plt.title = "before histogram equalization"
    plt.show()

    # sum value
    sumPix = 0

    # serializer data
    temp = {}
    for item in np.sort(image.flatten()):
        temp.setdefault(item, 0)
        temp[item] += 1
    for item in temp.keys():
        sumPix += temp[item] / imageTotalPixel
        sumValue = round(sumPix * 255)
        temp[item] = sumValue

    for i in range(height):
        for j in range(width):
            image[i][j] = temp[image[i][j]]

    plt.hist(image.flatten(), bins=255)
    plt.show()
    return image


image = cv2.imread("./images/lenna.png")
cv2.imshow("original", image)
result1 = HistogramEqualization(image[:, :, 0])
result2 = HistogramEqualization(image[:, :, 1])
result3 = HistogramEqualization(image[:, :, 2])
result = cv2.merge([result1, result2, result3])
cv2.imshow("hist equalization", result)

cv2.waitKey(0)
cv2.destroyAllWindows()