# Bilinear Interpolation
import cv2
import numpy as np


def bilinear_interpolation(img, times):
    scale = 1 / times

    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    res = np.zeros((np.floor(h * times).astype(np.int32), np.floor(w * times).astype(np.int32), c))

    for i in range(np.floor(h * times).astype(np.int32)):
        for j in range(np.floor(w * times).astype(np.int32)):
            for k in range(c):
                # geometric center symmetry
                src_x = (j + 0.5) * scale - 0.5
                src_y = (i + 0.5) * scale - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, h - 1)

                r1 = (src_x1 - src_x) * img[src_y0, src_x0, k] + (src_x - src_x0) * img[src_y0, src_x1, k]
                r2 = (src_x1 - src_x) * img[src_y1, src_x0, k] + (src_x - src_x0) * img[src_y1, src_x1, k]
                res[i, j, k] = int((src_y1 - src_y) * r1 + (src_y - src_y0) * r2)

    print(res.shape)
    return res


if __name__ == "__main__":
    src_img = cv2.imread("../img/lenna.png")
    res = bilinear_interpolation(src_img, 1.2)

    cv2.imshow("Src Image", src_img)
    cv2.imshow("Bilinear Interpolation", res.astype("uint8"))
    cv2.waitKey(0)
