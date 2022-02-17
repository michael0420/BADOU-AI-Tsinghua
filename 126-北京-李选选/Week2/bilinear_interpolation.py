#coding:utf8

'''
双线性插值
对于f(x+v,y+u)，确定其插值时使用不同的权重和f(x,y),f(x+1,y),f(x,y+1),f(x+1,y+1)的乘机来决定
f(i+u,j+v)=(1-u)*(1-v)*f(i,j)+(1-u)*v*f(i,j+1)+u*(1-v)*f(i+1,j)+u*v*f(i+1,j+1)
'''
import cv2
import numpy as np


def interpolation(img,size):
    h,w,channels=img.shape
    out_h,out_w=size[1],size[0]
    if h==out_h and w==out_w:
        return img.copy()
    out_img=np.zeros((out_h,out_w,3),np.uint8)
    scale_h=float(h)/out_h
    scale_w=float(w)/out_w
    for i in range(3):
        for _h in range(out_h):
            for _w in range(out_w):
                src_w=(_w+0.5)*scale_w-0.5
                src_h=(_h+0.5)*scale_h-0.5

                src_w0=int(np.floor(src_w))
                src_w1=min(src_w0+1,w-1)
                src_h0=int(np.floor(src_h))
                src_h1=min(src_h0+1,h-1)

                temp0=(src_w1-src_w)*img[src_h0,src_w0,i]+(src_w-src_w0)*img[src_h0,src_w1,i]
                temp1=(src_w1-src_w)*img[src_h1,src_w0,i]+(src_w-src_w0)*img[src_h1,src_w1,i]

                out_img[_h,_w,i]=int((src_h1-src_h)*temp0+(src_h-src_h0)*temp1)
    return out_img


img=cv2.imread("lenna.png")
out_img=interpolation(img,(1024,1024))
cv2.imshow("",out_img)
cv2.waitKey(0)
