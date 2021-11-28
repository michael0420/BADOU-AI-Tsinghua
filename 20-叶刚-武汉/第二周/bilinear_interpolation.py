"""
@author: GiffordY
Image scaling based on bilinear interpolation algorithm
"""
import numpy as np
import cv2 as cv


def bilinear_interpolation(src_img, dst_w: int, dst_h: int):
    ret = src_img.shape
    src_h = ret[0]
    src_w = ret[1]
    if src_w == dst_w and src_h == dst_h:
        return src_img.copy()
    if len(ret) == 3 and ret[2] == 3:
        dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    else:
        dst_img = np.zeros((dst_h, dst_w), dtype=np.uint8)
    ratio_x = src_w / dst_w
    ratio_y = src_h / dst_h
    for dst_y in range(0, dst_h):
        for dst_x in range(0, dst_w):
            # find the origin x and y coordinates of dst image x and y
            # use geometric center symmetry, if use direct way, src_x = dst_x * scale_x
            src_x = (dst_x + 0.5) * ratio_x - 0.5
            src_y = (dst_y + 0.5) * ratio_y - 0.5
            # find the coordinates of the points which will be used to compute the interpolation
            src_x1 = max(int(src_x), 0)
            src_y1 = max(int(src_y), 0)
            src_x2 = min(src_x1 + 1, src_w - 1)
            src_y2 = min(src_y1 + 1, src_h - 1)
            if 0 < dst_x < dst_w - 1 and 0 < dst_y < dst_h - 1:
                # The pixel value of the middle region is calculated by bilinear interpolation algorithm
                val_x_y1 = (src_x2 - src_x) * src_img[src_y1, src_x1] + (src_x - src_x1) * src_img[src_y1, src_x2]
                val_x_y2 = (src_x2 - src_x) * src_img[src_y2, src_x1] + (src_x - src_x1) * src_img[src_y2, src_x2]
                val_x_y = (src_y2 - src_y) * val_x_y1 + (src_y - src_y1) * val_x_y2
                dst_img[dst_y, dst_x] = val_x_y
            elif (dst_x, dst_y) == (0, 0) or (dst_x, dst_y) == (dst_w-1, 0) or (dst_x, dst_y) == (0, dst_h-1) or (dst_x, dst_y) == (dst_w-1, dst_h-1):
                # The pixel value at the corner is calculated by the nearest neighbor algorithm
                src_x = int(dst_x * ratio_x)
                src_y = int(dst_y * ratio_y)
                dst_img[dst_y, dst_x] = src_img[src_y, src_x]
            else:
                # The pixel value of the edge is calculated by linear interpolation algorithm
                if dst_y == 0 or dst_y == dst_h - 1:
                    dst_img[dst_y, dst_x] = (src_x2 - src_x) * src_img[src_y1, src_x1] + (src_x - src_x1) * src_img[src_y1, src_x2]
                else:
                    dst_img[dst_y, dst_x] = (src_y2 - src_y) * src_img[src_y1, src_x1] + (src_y - src_y1) * src_img[src_y2, src_x1]
    return dst_img


def main(img_path):
    img = cv.imread(img_path)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_dst = bilinear_interpolation(img, 600, 400)
    img_gray_dst = bilinear_interpolation(img_gray, 600, 400)
    cv.imshow('src image', img)
    cv.imshow('dst image', img_dst)
    cv.imshow('gray image', img_gray)
    cv.imshow('dst gary image', img_gray_dst)
    cv.imwrite('lenna_scaled.png', img_dst)
    cv.imwrite('lenna_gray_scaled.png', img_gray_dst)
    cv.waitKey(0)


if __name__ == '__main__':
    image_path = 'lenna.png'
    main(image_path)
