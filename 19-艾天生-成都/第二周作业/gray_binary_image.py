"""
@study id : 19
@author   : ai tian sheng
@project  : gray_binary_image
@note     : 灰度、二值化显示
"""
import numpy as np
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray

if __name__ == '__main__':
    # opencv读取，按任意键退出显示，并往下执行
    img = cv2.imread("lenna.png")
    h, w = img.shape[:2]
    img_gray = np.zeros([h, w], img.dtype)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
    print(img_gray)
    print("image show gray: %s" % img_gray)
    print("press any key exit pic show.\n")
    cv2.imshow("image show gray", img_gray)
    cv2.waitKey()

    # 显示原始图片
    img = plt.imread("lenna.png")
    plt.subplot(221)
    plt.imshow(img)
    # plt.show()
    print("---image lenna---")
    print(img)

    # 灰度化
    img_gray = rgb2gray(img)
    plt.subplot(222)
    plt.imshow(img_gray, cmap='gray')
    # plt.show()
    print("---image gray---")
    print(img_gray)

    # 二值化
    img_binary = np.where(img_gray >= 0.5, 1, 0)
    print("---image binary---")
    print(img_binary)
    print(img_binary.shape)
    plt.subplot(223)
    plt.imshow(img_binary, cmap='gray')
    # plt.show()

    # plot显示
    plt.show()
