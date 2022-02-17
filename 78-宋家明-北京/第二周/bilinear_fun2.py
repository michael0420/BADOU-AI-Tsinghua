import cv2
import numpy as np
import argparse

def bili_fun(img,dst_w,dst_h):
    
    src_h, src_w, _ = img.shape
    dst_img = np.zeros((dst_h,dst_w,3),np.uint8)
    
    dis_w, dis_h = src_w/dst_w, src_h/dst_h
    
    for ix in range(dst_w):
        for jy in range(dst_h):

            src_x = (ix+0.5)*dis_w-0.5
            src_y = (jy+0.5)*dis_h-0.5
        
            src_x0 = int(src_x)
            src_y0 = int(src_y)
            src_x1 = min(src_x0+1,src_w-1)
            src_y1 = min(src_y0+1,src_h-1)

            if src_x0<0 or src_y0<0:
                print(src_x0,'...',src_y0)

            temp0 = (src_x1-src_x)*img[src_y0,src_x0] + (src_x-src_x0)*img[src_y0,src_x1]
            temp1 = (src_x1-src_x)*img[src_y1,src_x0] + (src_x-src_x0)*img[src_y1,src_x1]
            
            dst_img[jy,ix] = (src_y1-src_y)*temp0 + (src_y-src_y0)*temp1

    return dst_img







if __name__=='__main__':

    img = cv2.imread('lenna.png')
    dst_w = 800
    dst_h = 800
    
    biliimg = bili_fun(img, dst_w, dst_h)
     
    print(biliimg.shape)
    cv2.imshow('bilinear',biliimg)
    cv2.waitKey(0)
