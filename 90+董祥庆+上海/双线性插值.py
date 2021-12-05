import cv2
import time
from math import ceil, floor
import numpy as np


def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=np.uint8)  # 创建一个数组，通过往里面填值，形成新的图片
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h  # 计算 dst与src的比例系数
    for i in range(channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 计算dst(x,y)对应回源图src的哪个坐标
                # 使用几何中心对称的方法
                # 如果不使用几何中心对成就写成： src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 找出用于插值的四个邻近点坐标，(x0,y0),(x0,y1),(x1,y0),(x1,y1)
                src_x0 = int(floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)  # 与边界点比，取小的一个,减1是因为是从0开始算的
                src_y0 = int(floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)  # 与边界点比，取小的一个

                temp0 = ((src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i])
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
    return dst_img


if __name__ == '__main__':
    img = cv2.imread(r'/Users/oh/Desktop/zuoye/WechatIMG13.jpeg')
    start = time.time()
    dst = bilinear_interpolation(img, (200, 200))
    print('cost {} seconds'.format(time.time() - start))
    cv2.imshow('result', dst)
    cv2.waitKey(0)


