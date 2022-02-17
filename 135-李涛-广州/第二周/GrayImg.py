import cv2
import matplotlib.pyplot as plt
import numpy as np

'''
    1,23,3
    2,3,4
'''


def test3D():
    data_array = np.zeros((3, 4, 5), dtype=int)
    data_array[1, 2, 3] = 1
    h, w = data_array.shape[:2]
    print(h, w)
    print(data_array.shape)


def img2gray(img):
    h, w = img.shape[:2]
    img_gray = np.zeros([w, h], dtype=np.float)
    print(img.shape, h, w)
    for i in range(h):
        for j in range(w):
            m = img[i, j]
            img_gray[i, j] = (m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)/256
    return img_gray


def two_value_gray(img_gray):
    print(img_gray)
    h, w = img.shape[:2]
    for i in range(h):
        for j in range(w):
            if img_gray[i, j] >= 0.5:
                img_gray[i, j] = 1
            else:
                img_gray[i, j] = 0
    return img_gray


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.subplot(221)
    plt.imshow(img2)
    img_gray = img2gray(img)
    print('gray_data_array')
    # print(img_gray)
    plt.subplot(222)
    plt.imshow(img_gray,cmap='gray')
    img_gray = two_value_gray(img_gray)
    plt.subplot(223)
    plt.imshow(img_gray,cmap='gray')
    plt.show()
