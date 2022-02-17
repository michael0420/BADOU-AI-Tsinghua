import cv2
import numpy as np


def binary(img, threshold):
    h = img.shape[0]
    w = img.shape[1]
    res = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if img[i, j][0] / 3 + img[i, j][1] / 3 + img[i, j][2] / 3 < threshold:
                res[i, j] = 0
            else:
                res[i, j] = 255

    return res


if __name__ == '__main__':
    src_img = cv2.imread("../img/lenna.png")
    binary = binary(src_img, 130)
    cv2.imshow("binary image", binary.astype("uint8"))
    cv2.waitKey(0)
