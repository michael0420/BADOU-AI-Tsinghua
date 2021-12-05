import cv2
import numpy as np


def bilinear_interpolation(img,out_dim):
    sh,sw,sc = img.shape
    dh,dw =out_dim[0],out_dim[1]

    if dh == sh and dw == sw:
        return img.copy()

    scalex=float(sw)/dw
    scaley=float(sh)/dh
    dst_img = np.zeros((dh,dw,3),dtype=np.uint8)
    for i in range(3):        #如果需要多个像素参与计算，就必须逐通道进行计算
        for dst_y in range(dh):
            for dst_x in range(dw):
                sx0=(dst_x+0.5)*scalex-0.5      #计算对应原图像素位置
                sy0=(dst_y+0.5)*scaley-0.5

                sx1=int(np.floor(sx0))      #找到参与计算的四个像素位置  一会试下不转换为整型或者不用np.floor
                sx2=min(sx1+1,sw-1)          #防止边缘处溢出
                sy1=int(np.floor(sy0))
                sy2=min(sy1+1,sh-1)

                dst_img[dst_y,dst_x,i]=int((sy0-sy1)*(sx0-sx1)*img[sy2,sx2,i]+(sy0-sy1)*(sx2-sx0)*img[sy2,sx1,i]+(sy2-sy0)*(sx0-sx1)*img[sy1,sx2,i]+(sx2-sx0)*(sy2-sy0)*img[sy1,sx1,i])

    return dst_img

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('lenna',img)
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()




