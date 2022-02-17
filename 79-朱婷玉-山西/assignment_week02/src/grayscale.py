import cv2
import numpy as np


def gray_scale(img):
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    gray = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            for k in range(c):
                gray[i, j] += img[i][j][k] / 3

    return gray


if __name__ == "__main__":
    src_img = cv2.imread("../img/lenna.png")
    gray_scale = gray_scale(src_img)
    cv2.imshow("Grayscale", gray_scale.astype("uint8"))
    cv2.waitKey(0)
