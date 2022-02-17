"""
@study id : 19
@author   : ai tian sheng
@project  : nearest_interpolation_image
@note     : 临近插值算法
"""
import cv2.cv2 as cv2
import numpy as np

def func_zoom(_img, _out_dim):
    """
    临近插值函数
    :param _img:
    :param _out_dim:
    :return:
    """
    height, width, channels = _img.shape
    src_h, src_w = _out_dim[0], _out_dim[1]
    emptyImage = np.zeros((src_h, src_w, channels), np.uint8)
    sh = src_h/height
    sw = src_w/width
    for i in range(src_h):
        for j in range(src_w):
            x = int(i/sh)
            y = int(j/sw)
            emptyImage[i,j] = img[x,y]
    return emptyImage


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    zoom = func_zoom(img, (800,800))
    #print(zoom)
    print(img.shape)
    print(zoom.shape)
    #显示原图
    cv2.imshow("original image", img)
    #显示插值放大后的图
    cv2.imshow("neast interpolation image", zoom)
    cv2.waitKey(0)
