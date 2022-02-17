"""
@author: Dong Chen
@time: 01/11/2022
@Reference: Teacher Wang's code

do image bilinear interpolation
"""

import numpy as np
import cv2

#python implementation of bilinear interpolation
def bilinear_interpolation(img,dst_size):
    #get source image's heigth, width and channels
    src_h, src_w, channels = img.shape
    dst_h, dst_w = dst_size[1], dst_size[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    #set destination image size
    dst_img = np.zeros((dst_h, dst_w, 3), img.dtype)
    ratio_x, ratio_y = float(src_w)/dst_w, float(src_h)/dst_h
    for i in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #find the origin x and y coordinates of dst image x and y
                #use geometric center symmetry
                src_x = (dst_x + 0.5) * ratio_x - 0.5
                src_y = (dst_y + 0.5) * ratio_y - 0.5

                #find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0+1, src_h - 1)

                #calculate the interpolation
                #x axis interpolation
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                #y axis interpolation
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if __name__ == '__main__':
    img=cv2.imread("lenna.png")
    dst_size = (700, 700)
    dst_img = bilinear_interpolation(img,dst_size)
    cv2.imshow("original image", img)
    cv2.imshow("bilinear interpolation", dst_img)
    cv2.waitKey(0)
    cv2.destroyWindow()
