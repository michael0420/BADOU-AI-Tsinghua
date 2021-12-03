import cv2
import numpy as np


def interpolation(img, size):
    """
    convert img to new img with the nearest interpolation
    :param img: the origin img
    :param size: the size of new img
    :return: img
    """
    height, width, channels = img.shape
    h = size / height
    w = size / width
    return np.array([[img[int(i/h), int(j/w)] for j in range(size)] for i in range(size)])
    # new_img = np.zeros([size, size, channels], np.uint8)
    # for i in range(size):
    #     for j in range(size):
    #         new_img[i, j] = img[int(i/h), int(j/w)]
    # return new_img


# import image from file
im = cv2.imread("lenna.png")
n_img = interpolation(im, 800)
print(n_img.shape)
print("image show n_img: %s" % n_img)
cv2.imshow("nearest interpolation", n_img)
cv2.waitKey(0)
