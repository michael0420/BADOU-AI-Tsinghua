#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： xufeng
# datetime： 2021/11/23 23:23 
# ide： PyCharm

import cv2
import numpy as np


def bilinear_interplot(img, out_dim):
    src_h, src_w, channels = img.shape
    dst_h, dst_w = out_dim[0], out_dim[1]
    print("原图高 = {}，原图宽 = {}".format(src_h, src_w))
    print("目标图高 = {}， 目标图宽 = {}".format(dst_h, dst_w))

    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):

                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                # srcX = dstX * (srcWidth / dstWidth) + 0.5 * (srcWidth / dstWidth - 1)
                # srcY = dstY * (srcHeight / dstHeight) + 0.5 * (srcHeight / dstHeight - 1)

                src_x = dst_x * scale_x + 0.5 * (scale_x - 1) #14.5
                src_y = dst_y * scale_y + 0.5 * (scale_y - 1) # 20.2

                #找到原始图中对应的四个参与运算的点的坐标
                src_x0 = int(np.floor(src_x)) # 14
                src_x1 = min(src_x0 + 1, src_w - 1) #15
                src_y0 = int(np.floor(src_y)) #20
                src_y1 = min(src_y0 + 1, src_h - 1) #21

                #计算插值
                tmp_0 = (src_x - src_x0) * img[src_y0, src_x1, i] + (src_x1 - src_x) * img[src_y0, src_x0, i]
                tmp_1 = (src_x - src_x0) * img[src_y1, src_x1, i] + (src_x1 - src_x) * img[src_y1, src_x0, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * tmp_0) + int((src_y - src_y0) * tmp_1)
    return dst_img

if __name__ == '__main__':
    img = cv2.imread("../../../BaiduNetdiskDownload/lenna.png")
    dst = bilinear_interplot(img, (700, 700))
    cv2.imshow('img', img)
    cv2.imshow("bilinear", dst)
    cv2.waitKey(0)