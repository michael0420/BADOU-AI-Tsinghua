from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def calc_histogram(img):
    hist = np.zeros(256)

    for j in range(img.shape[0]):
        for i in range(img.shape[1]):
            tmp = img[i][j]
            hist[tmp] = hist[tmp] + 1
            #print("tmp = {0}, hist = {1}".format(tmp, hist[tmp]))

    return hist

def equalize_histogram(img, hist):
    equalize_hist_img = np.zeros(img.shape)
    transform_hist = np.zeros(256)
    sumPi = 0
    total_pix = img.shape[0] * img.shape[1]

    for i in range (256):
        Pi = hist[i] / total_pix
        sumPi = sumPi + Pi
        tmp = sumPi * 256 - 1
        if tmp < 0:
            transform_hist[i] = 0
        else:
            transform_hist[i] = int(tmp)

        print("transfor:{}-{}".format(i, transform_hist[i]))


    for j in range (img.shape[0]):
        for i in range(img.shape[1]):
            tmp = img[i][j]
            #print("{} - {}".format(tmp, transform_hist[tmp]))
            equalize_hist_img[i][j] = transform_hist[tmp] / 255

    return equalize_hist_img

img = cv2.imread("lenna.png")

# Test 1:灰度直方图统计及直方图均衡化

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #call opencv
#
h, w = img.shape[0], img.shape[1]
hist = calc_histogram(img_gray)
img_equa_img = equalize_histogram(img_gray, hist)
plt.figure()#新建一个图像
plt.title("Grayscale Histogram")
plt.xlabel("Bins")#X轴标签
plt.ylabel("# of Pixels")#Y轴标签
plt.plot(hist)
plt.xlim([0,256])#设置x坐标轴范围
plt.show()

cv2.imshow("equalize img", img_equa_img)
cv2.waitKey(0)

