import cv2 as cv2
import numpy as np

def DIYModle(M,src_r,src_c,src_ch):
    #找四个点
    x0,y0 = int(np.floor(src_r)),int(np.floor(src_c))
    x1,y1 = min(x0+1,M.shape[0]-1),min(y0+1,M.shape[1]-1)

    # 取四个点位置对应的值
    pixel_lt,pixel_rt,pixel_lb,pixel_rb = M[x0][y0][src_ch],M[x0][y1][src_ch],M[x1][y0][src_ch],M[x1][y1][src_ch]

    # 插值，2次x方向，1次y方向
    pixel_mt = (y1-src_c)*pixel_lt+(src_c-y0)*pixel_rt
    pixel_mb = (y1-src_c)*pixel_lb+(src_c-y0)*pixel_rb

    pixel = int((x1-src_r)*pixel_mt+(src_r-x0)*pixel_mb)

    return pixel

src_img = cv2.imread('lenna.png')
dst_img = np.zeros((800,800,3),dtype=np.uint8)
rows,cols,channels = src_img.shape
dst_row,dst_col,dst_channels = 800,800,3
# 比例系数
factor_x ,factor_y= float(rows) / dst_row,float(cols) /dst_col
for channle in range(dst_channels):
    for row in range(dst_row):
        for col in range(dst_col):
            # 重新对其中心
            src_x = (row + 0.5)*factor_x-0.5
            src_y = (col + 0.5)*factor_y-0.5
            # 赋值
            dst_img[row,col,channle] = DIYModle(src_img,src_x,src_y,channle)

cv2.imshow('DIY',dst_img)
cv2.waitKey()

