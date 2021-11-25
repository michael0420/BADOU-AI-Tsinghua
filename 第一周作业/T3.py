# -*- coding: utf-8 -*-
'''
@Time    : 2021/11/21/00021 21:13
@Author  : Chen Quan
@FileName: T3.py
@Software: PyCharm
'''
'''
3.双线性插值实现
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt


def transform(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bilinear_interpolation(img,out_dim):
    src_h,src_w,channel=img.shape
    dst_h,dst_w,=out_dim[1],out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h==dst_h and src_w==dst_w:
        return img.copy()

    dst_img=np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    scale_x,scale_y=float(src_w)/dst_w,float(src_h)/dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x=(dst_x+0.5)*scale_x-0.5
                src_y=(dst_y+0.5)*scale_y-0.5

                src_x1=int(np.floor(src_x))
                src_x2=min(src_x1+1,src_w-1)
                src_y1=int(np.floor(src_y))
                src_y2=min(src_y1+1,src_h-1)


                temp0=(src_x2-src_x)*img[src_y1,src_x1,i]+(src_x-src_x1)*img[src_y1,src_x2,i]
                temp1=(src_x2-src_x)*img[src_y2,src_x1,i]+(src_x-src_x1)*img[src_y2,src_x2,i]

                dst_img[dst_y,dst_x,i]=int((src_y2-src_y)*temp0+(src_y-src_y1)*temp1)

    print(dst_img)

    return dst_img
if __name__ == '__main__':
    img=cv2.imread("lenna.png")
    img_origin=transform(img)
    dst=bilinear_interpolation(img,(700,700))
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False
    plt.subplot(221)
    plt.title("BGR")
    plt.imshow(img)

    plt.subplot(222)
    plt.title("RGB")
    plt.imshow(img_origin)

    plt.subplot(223)
    plt.title("插值后BGR",fontsize=10)
    plt.imshow(dst)

    plt.subplot(224)
    plt.title("插值后RGB",fontsize=10)
    plt.imshow(transform(dst))
    plt.show()

