import cv2
import numpy as np


def hist_equl(img):
    h, w = img.shape[0:2]
    size = h * w
    sum_p = 0

    new_img = np.zeros_like(img, dtype=img.dtype)

    for i in range(256):
        img_i_index = img == i
        num_count = np.count_nonzero(img_i_index)
        p = num_count / size
        sum_p += p

        value = np.round(sum_p * 256 - 1)
        value = value if value > 0 else 0
        new_img[img_i_index] = value

    return new_img


if __name__ == '__main__':
    img_path = "../pics/lenna.png"
    img_bgr = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_dst = hist_equl(img_gray)

    cv2.imshow("img_src", img_gray)
    cv2.imshow("img_dst", img_dst)
    cv2.waitKey(0)
