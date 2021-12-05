'''
@author:Jelly

将图像进行上采样放大处理，采用双线性插值方法

相关接口：

def Bilinear_Interp(img,hight=800,wight=800):
函数用途：对图像进行双线性插值

'''
import cv2
import numpy as np

def Bilinear_Interp(img,hight=800,wight=800):
    '''
    对输入图像进行双线性插值
    :param img: 输入图像数据
    :param hight: 要扩大的图片高度
    :param wight: 要扩大的图片宽度
    :return: 插值后的图片数据
    '''
    src_h,src_w,channel = img.shape
    dst_h,dst_w = hight,wight
    if src_h>dst_h or src_w>dst_w:
        print('输入图片维度错误，但前图片维度为hight=%d,wight=%d。',hight,wight)
        return img
    dst_img = np.zeros((hight,wight,channel),img.dtype)
    scale_x,scale_y = float(src_h)/dst_h,float(src_w)/dst_w,
    for i in range(channel):
        for dst_x in range(dst_h):
            for dst_y in range(dst_w):
                #让输出图与原图的几何中心重合
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                #像素边界选择
                src_x0 = int(np.floor(src_x))  #np.floor对多维元素向下取整
                src_x1 = min(src_x0 + 1,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1,src_h - 1)

                #计算双线性插值的结果
                fr1 = (src_x1 - src_x) * img[src_x0,src_y0,i] + (src_x - src_x0) * img[src_x1,src_y0,i]
                fr2 = (src_x1 - src_x) * img[src_x0,src_y1,i] + (src_x - src_x0) * img[src_x1,src_y1,i]
                dst_img[dst_x,dst_y,i] = int((src_y1 - src_y) * fr1 + (src_y - src_y0) * fr2)

    return dst_img

img = cv2.imread('lenna.png')
dst = Bilinear_Interp(img,800,1000)
cv2.imshow('bilinear interp',dst)
cv2.waitKey()