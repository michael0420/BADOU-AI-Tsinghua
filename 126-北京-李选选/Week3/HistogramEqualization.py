# coding:utf8

'''
直方图均衡化
1.Cv2.equalizeHist()
2.自实现
'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

# # cv2.equalizeHist得到均衡化后的图像
# img=cv2.imread('lenna.png')
# gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# dst=cv2.equalizeHist(gray)
# # cv2.imshow("1",gray)
# # cv2.imshow("2",dst)
# cv2.imshow("Compare",np.hstack([gray,dst]))
# cv2.waitKey(0)
# hist=cv2.calcHist([dst],[0],None,[256],[0,256])
# plt.figure()
# plt.plot(hist)
# plt.show()
#
# #彩色图均衡化
# channels=cv2.split(img)
# dst_b=cv2.equalizeHist(channels[0])
# dst_g=cv2.equalizeHist(channels[1])
# dst_r=cv2.equalizeHist(channels[2])
#
# dst=cv2.merge((dst_b,dst_g,dst_r))
# cv2.imshow("Compare",np.hstack([img,dst]))
# cv2.waitKey(0)
#


'''
自实现
1.求出原图的灰度直方图
2.求累加直方图
3.套用公式，得到p与q的对应关系，我这里用字典存储
4.遍历灰度图，将该位置的灰度替换成字典对应key的value值
'''


def equalizeHist_diy(img):
    src = cv2.calcHist([img], [0], None, [256], [0, 256])
    size = img.shape[0] * img.shape[1]
    total = np.zeros([256, 1], float)
    dict = {}
    total[0, 0] = src[0, 0]
    for index in range(1, src.shape[0]):
        total[index, 0] = total[index - 1, 0] + src[index, 0]
        dict[index] = round(total[index, 0] / size * 256 - 1)
    dst = np.zeros([img.shape[0], img.shape[1]], np.uint8)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            dst[r, c] = dict.get(img[r, c])
    return dst


'''
直接利用直方图求累加直方图，简化过程
需要注意的是hist是2元数组，因此访问时需要用hist[x,y]访问！！！
'''


def diy2(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist[0] = hist[0]
    for index in range(1, hist.shape[0]):
        hist[index, 0] += hist[index - 1, 0]

    dst = np.zeros([img.shape[0], img.shape[1]], np.uint8)

    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            dst[r, c] = round(hist[img[r, c], 0] / img.size * 256 - 1)
    return dst


lenna = cv2.imread("lenna.png")
gray = cv2.cvtColor(lenna, cv2.COLOR_BGR2GRAY)

img1 = diy2(gray)
cv2.imshow("1", img1)

# img2=equalizeHist_diy(gray)
# cv2.imshow("2",img2)
cv2.waitKey(0)

# dst_gray=equalizeHist_diy(gray)
# gray1=cv2.equalizeHist(gray)
# # gray2=diy2(gray)
# cv2.imshow("gray",np.hstack((gray,gray1,dst_gray)))
#
# channels=cv2.split(lenna)
# dst_b=equalizeHist_diy(channels[0])
# dst_g=equalizeHist_diy(channels[1])
# dst_r=equalizeHist_diy(channels[2])
# dst=cv2.merge((dst_b,dst_g,dst_r))
#
# b=cv2.equalizeHist(channels[0])
# g=cv2.equalizeHist(channels[1])
# r=cv2.equalizeHist(channels[2])
# dst1=cv2.merge((b,g,r))
#
# cv2.imshow("dst",np.hstack((lenna,dst1,dst)))
# cv2.waitKey(0)
