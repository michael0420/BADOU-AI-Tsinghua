"""
最邻近插值处理图片
"""
import numpy as np
import cv2
def nearestInsert(imag, height_tar, weight_tar):
    height, weight, channels = imag.shape
    dtype = imag.dtype
    imag_new = np.zeros([weight_tar, height_tar, channels], dtype)
    scale_x = weight_tar/weight
    scale_y = height_tar/height
    for i in range(weight_tar):
        for j in range(height_tar):
            # x = int(i/scale_x)
            # y = int(j/scale_y)
            x = int(round(i/scale_x, 0))
            y = int(round(j/scale_y, 0))
            imag_new[i, j] = imag[x, y]
    return imag_new
if __name__ == '__main__':
    imag = cv2.imread("lenna.png")
    print(imag.shape[:2])
    imag2 = nearestInsert(imag, 700, 700)
    print(imag2.shape)
    cv2.imshow("origin_picture", imag)
    cv2.imshow("nearestInsert_picture", imag2)
    cv2.waitKey(0)
