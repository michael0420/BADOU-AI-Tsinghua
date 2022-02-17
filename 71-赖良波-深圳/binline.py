#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

'''
python implementation of bilinear interpolation
'''

def billine_near(img,out_dim):
    src_h,src_w,channel=img.shape
    dest_h,dest_w=out_dim[0],out_dim[1]
    # 大写一样， 直接复制图片返回
    if src_h==dest_h and src_w==dest_w:
        return img.copy()
    #新建图片
    new_img=np.zeros([dest_h,dest_w,channel],img.dtype)
    scale_x= float(src_h)/dest_h
    scale_y = float(src_w) / dest_w

    for i in range(3):
        for dest_y in range(dest_h):
            for dest_x in range(dest_w):

                src_x = (dest_x + 0.5) * scale_x - 0.5
                src_y = (dest_y + 0.5) * scale_y - 0.5

                src_x = (dest_x + 0.5) * scale_x - 0.5
                src_y = (dest_y + 0.5) * scale_y - 0.5

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0+1,src_w-1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                #R1=((src_x1-src_x)/(src_x1-src_x0))*img[src_y0,src_x0,i]+((src_x-src_x0)/(src_x1-src_x0))*img[src_y0,src_x1,i]
                R1 = (src_x1 - src_x)  * img[src_y0, src_x0, i] \
                     + (src_x - src_x0) * img[src_y0, src_x1, i]
                #R2 = (src_x1 - src_x) / (src_x1 - src_x0) * img[src_y1, src_x0, i] + (src_x - src_x0) / (src_x1 - src_x0) * img[src_y1, src_x1, i]
                R2 = (src_x1 - src_x) * img[src_y1, src_x0, i] \
                     + (src_x - src_x0) * img[src_y1, src_x1, i]
                P=(src_y1-src_y)*R1\
                   +(src_y-src_y0)*R2

                new_img[dest_y, dest_x, i]=P


    return new_img

if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    dest_img=billine_near(img,(800,800))
    cv2.imshow("dest_img_title",dest_img)
    cv2.waitKey()



