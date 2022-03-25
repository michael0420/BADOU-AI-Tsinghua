"""
author:xswwhy
双线性插值
"""
import cv2
import numpy as np


def bilinear_interp(src_img, dst_h, dst_w):
    src_h, src_w, c = src_img.shape
    dst_img = np.zeros([dst_h, dst_w, c], src_img.dtype)
    scale_h = dst_h / src_h
    scale_w = dst_w / src_w
    for i in range(dst_h):
        for j in range(dst_w):
            # 这两行区别于最邻近插值,不用int,保留float
            h = i / scale_h
            w = j / scale_w

            offset_h, offset_w = 1, 1
            # 这里映射到src_img最下面或者最右边的像素的时候,后面+1会越界,保护一下
            if int(h + 1) == src_h:
                offset_h = 0
            if int(w + 1) == src_w:
                offset_w = 0
            # 定位到src_img的四个像素
            pixel_1 = src_img[int(h), int(w)]
            pixel_2 = src_img[int(h), int(w) + offset_w]
            pixel_3 = src_img[int(h + offset_h), int(w)]
            pixel_4 = src_img[int(h + offset_h), int(w) + offset_w]

            # 双线性插值实际上是计算三次单线性插值
            pixel_m1 = count_new_pixel_by_weight(pixel_1, pixel_2, w)
            pixel_m2 = count_new_pixel_by_weight(pixel_3, pixel_4, w)
            pixel_m3 = count_new_pixel_by_weight(pixel_m1, pixel_m2, h)
            dst_img[i, j] = pixel_m3
    return dst_img


def count_new_pixel_by_weight(pixel_1, pixel_2, w):
    weight_later = w - int(w)
    weight_front = 1 - weight_later
    result_pixel = pixel_1 * weight_front + pixel_2 * weight_later
    return result_pixel.astype(np.uint8)


img_bgr = cv2.imread("lenna.png")
img_large = bilinear_interp(img_bgr, 1000, 1000)
img_small = bilinear_interp(img_bgr, 200, 200)

# show
cv2.imshow("original", img_bgr)
cv2.imshow("large", img_large)
cv2.imshow("small", img_small)
cv2.waitKey(0)
