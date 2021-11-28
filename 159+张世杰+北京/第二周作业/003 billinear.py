# encoding: utf-8
import numpy as np
import cv2
"""
双线性差值
"""
def bilinear_img(img, out_dim):
    src_h, src_w, src_ch = img.shape  # 获取输入图片的长、宽、通道数；
    dst_h, dst_w = out_dim[1], out_dim[0]  # 获得输出图像的高、宽
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    # 如果输出尺寸不变， 直接复制图片
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_h, 3), dtype=np.uint8)  # 生成一个空白图像
    # 此时的比例为缩小比例， 因为双线性插值的原理是将dst_img按比例缩小值src图像后计算像素点
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                src_x = (dst_x + 0.5) * scale_x - 0.5  # 进行中心对称：1/2 *（1-缩小比例）
                src_y = (dst_y + 0.5) * scale_y - 0.5  # 进行中心对称：1/2 *（1-缩小比例）
                src_x0 = int(np.floor(src_x))  # 确定 x0 值
                src_x1 = min(src_x0 + 1, src_w - 1)  # 确定 x1 值 ，此处要进行最小值处理， 避免越界
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_img(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
