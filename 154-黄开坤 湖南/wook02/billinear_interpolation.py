#coding:utf-8

import numpy as np
import cv2

'''python implementaton of bilinear interpolation'''


def bilinearInterp(img, out_img_dim):
    src_h, src_w, chan = img.shape
    dst_h, dst_w = out_img_dim[1], out_img_dim[0]
    print('src_h, src_w:', src_h, src_w)        #(512 512)
    print('dst_h, dst_w:', dst_h, dst_w)        #(700 700)
    if src_h ==dst_h and src_w == dst_w:        #判断大小是否相等
        return img.copy()
    dst_img_empty = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w)/dst_w, float(src_h)/dst_h   #(srcWidth/dstWidth),(srcHeight/dstHeight)
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #中心对齐
                #SrcX + 0.5 = (dstX+0.5)*(srcWidth/dstWidth)
                #SrcY + 0.5 = (dstY+0.5)*(srcHeight/dstHeight)
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                #找到四个点的坐标
                src_x0 = int(np.floor(src_x))   #np.floor()返回不大于输入参数的最大整数。（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)     # 防止超出图片
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                #calculate the interpolation
                #f(R1)=(x2-x)*f(Q11) + (x-x1)*f(Q21)
                R1 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                #f(r2) = (x2-x)*f(Q12) + (x-x1)*f(Q22)
                R2 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # f(x, y)=(y2-y)f(R1) + (y-y1)*f(R2)
                dst_img_empty[dst_y, dst_x, i] = int((src_y1 - src_y) * R1 + (src_y - src_y0) * R2)

    return dst_img_empty

if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinearInterp(img, (700, 700))
    cv2.imshow('bilinear interpolation:', dst)
    cv2.imshow('image', img)          #原图
    cv2.waitKey(0)