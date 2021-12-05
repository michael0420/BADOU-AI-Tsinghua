import cv2
import numpy as np


def bilinear_interpolation(img, dst_size):
    dst_h, dst_w = dst_size
    # 原始尺寸
    if len(img.shape) > 2:
        src_h, src_w, channel = img.shape
    else:
        src_h, src_w = img.shape
        channel = 1

    # 缩放比率: 原图/输出
    ratio_h = src_h / dst_h
    ratio_w = src_w / dst_w

    # 目标图像坐标矩阵
    y_dst_index = np.arange(dst_h, dtype=np.int32)
    x_dst_index = np.arange(dst_w, dtype=np.int32)

    # 对应原图坐标
    y_src_index = (y_dst_index + 0.5) * np.array(ratio_h) - 0.5
    x_src_index = (x_dst_index + 0.5) * np.array(ratio_w) - 0.5

    x_src_index = np.tile(x_src_index, dst_h)
    y_src_index = y_src_index.repeat(dst_w)

    # 原图坐标较小点
    y_src_min_index = np.int32(np.floor(y_src_index))
    x_src_min_index = np.int32(np.floor(x_src_index))

    # 原图坐标较大点
    y_src_max_index = y_src_min_index + 1
    x_src_max_index = x_src_min_index + 1
    y_src_max_index[y_src_max_index >= src_h] = src_h - 1
    x_src_max_index[x_src_max_index >= src_w] = src_w - 1

    # 计算插值
    temp_up = np.multiply(img[y_src_min_index, x_src_min_index].T, (x_src_max_index - x_src_index)) \
              + np.multiply(img[y_src_min_index, x_src_max_index].T, (x_src_index - x_src_min_index))
    temp_down = np.multiply(img[y_src_max_index, x_src_min_index].T, (x_src_max_index - x_src_index)) \
                + np.multiply(img[y_src_max_index, x_src_max_index].T, (x_src_index - x_src_min_index))

    # 赋值给目标大小图像
    img_dst = np.int32(np.multiply(y_src_max_index - y_src_index, temp_up) \
                       + np.multiply(y_src_index - y_src_min_index, temp_down))
    img_dst = np.uint8(img_dst.T)
    img_dst = img_dst.reshape(dst_h, dst_w, channel)

    return img_dst


if __name__ == '__main__':
    img_path = "../pics/lenna.png"
    img_bgr = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_dst = bilinear_interpolation(img_bgr, (700, 700))
    cv2.imshow("img_src", img_bgr)
    cv2.imshow("img_resized", img_dst)
    cv2.waitKey(0)
