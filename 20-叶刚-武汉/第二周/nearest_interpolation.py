"""
@author: GiffordY
Image scaling based on nearest neighbor interpolation algorithm
"""
import numpy as np
import cv2 as cv


def nearest_interpolation(src_img, dst_w: int, dst_h: int):
    """
    nearest neighbor interpolation algorithm
    :param src_img: input image
    :param dst_w: width of dst_image
    :param dst_h: height of dst_image
    :return: dst_image: scaled image
    """
    ret = src_img.shape
    src_h, src_w = ret[:2]
    if src_h == dst_h and src_w == dst_w:
        return src_img.copy()
    ratio_h = src_h / dst_h
    ratio_w = src_w / dst_w
    if len(ret) == 2:   # grayscale image
        dst_img = np.zeros((dst_h, dst_w), dtype=np.uint8)
    else:   # color image
        dst_img = np.zeros((dst_h, dst_w, ret[2]), dtype=np.uint8)
    for col in range(dst_w):
        for row in range(dst_h):
            src_x = int(col * ratio_w)
            src_y = int(row * ratio_h)
            dst_img[row, col] = src_img[src_y, src_x]
    return dst_img


def main(img_path):
    img = cv.imread(img_path)
    # zoom image by my function
    img_dst = nearest_interpolation(img, 600, 400)
    # zoom image by opencv function
    img_zoom = cv.resize(img, (600, 400), interpolation=cv.INTER_NEAREST)
    # calculate the difference of results of two methods
    sub_img = cv.subtract(img_dst, img_zoom)
    print("Max value of difference of two images: ", np.max(sub_img))

    cv.imshow('src image', img)
    cv.imshow('dst image', img_dst)
    cv.imshow('zoomed image', img_zoom)
    cv.waitKey(0)


if __name__ == '__main__':
    image_path = 'lenna.png'
    main(image_path)
