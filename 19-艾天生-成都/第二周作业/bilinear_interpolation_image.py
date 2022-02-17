"""
@study id : 19
@author   : ai tian sheng
@project  : bilinear_interpolation_image
@note     : 双线插值算法
"""
import numpy as np
import cv2.cv2 as cv2

def bilinear_interpolation(_img, _out_dim):
    """
    双线插值计算
    :param _img:
    :param _out_dim:
    :return:
    """
    src_h, src_w, channel = _img.shape
    dst_h, dst_w = _out_dim[1], _out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)

    #创建3通道缓存
    dst_img = np.zeros((dst_h, dst_w, 3), dtype = np.uint8)
    scale_x, scale_y = float(src_w/dst_w), float(src_h/dst_h)
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #计算放大后的新坐标与原图坐标之间对应关系（中心对称的坐标点）
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5
                #找到src_x、src_y的临近坐标点计算
                src_x0 = int(np.floor(src_x)) #得到小于src_x的整数最大值
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y)) #得到小于src_y的整数最大值
                src_y1 = min(src_y0 + 1, src_h - 1)
                #计算像素值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y)*temp0 + (src_y - src_y0)*temp1)
    return dst_img

if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    dst = bilinear_interpolation(img, (1000, 800))
    cv2.imshow("orginal image", img)
    cv2.imshow("bilinear interp", dst)
    cv2.waitKey(0)