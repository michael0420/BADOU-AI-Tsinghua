# Nearest Neighbor Interpolation
import cv2
import numpy as np


def nearest_neighbor(img, times):
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    res = np.zeros((np.floor(h * times).astype(np.int32), np.floor(w * times).astype(np.int32), c))

    for i in range(np.floor(h * times).astype(np.int32)):
        for j in range(np.floor(w * times).astype(np.int32)):
            for k in range(c):
                res[i][j][k] = img[int(i / times), int(j / times)][k]

    print(res.shape)
    return res


if __name__ == "__main__":
    src_img = cv2.imread("../img/lenna.png")
    nearest = nearest_neighbor(src_img, 2.3)
    cv2.imshow("src img", src_img)
    cv2.imshow("Nearest Neighbor", nearest.astype("uint8"))
    cv2.waitKey(0)
