import cv2
import numpy as np


def img2gray(img, mode=0):
    img_array = np.array(img)
    img_b = img_array[:, :, 0]
    img_g = img_array[:, :, 1]
    img_r = img_array[:, :, 2]

    if mode == 1:
        img_gray = np.uint8(np.array(76 * np.int32(img_r) + 151 * np.int32(img_g) + 28 * np.int32(img_b)) >> 8)

    elif mode == 2:
        img_gray = np.uint8(np.array(30 * np.int32(img_r) + 59 * np.int32(img_g) + 11 * np.int32(img_b)) / 100)

    else:
        img_gray = np.uint8(0.3 * img_r + 0.59 * img_g + 0.11 * img_b)

    cv2.imshow("gray", img_gray)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = "../pics/lenna.png"
    img_bgr = cv2.imread(img_path)
    img2gray(img_bgr, 0)
