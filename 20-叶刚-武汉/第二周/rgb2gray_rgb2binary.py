"""
@author: GiffordY
Convert color image to a grayscale image or a binary image
"""
import numpy as np
import cv2 as cv


def rgb2gray(src_img):
    """
    Convert the input color image to a grayscale image.
    If the input image is a grayscale image, return directly
    :param src_img: Input image
    :return: Output grayscale image
    """
    ret = src_img.shape
    rows, cols = ret[:2]
    if len(ret) != 3:
        return src_img
    else:
        dst_img = np.zeros([rows, cols], dtype=src_img.dtype)
        for i in range(rows):
            for j in range(cols):
                color_val = src_img[i, j]
                dst_img[i, j] = color_val[2] * 0.3 + color_val[1] * 0.59 + color_val[0] * 0.11
        return dst_img


def gray2binary(src_img, threshold=128):
    """
    Convert the input image to a binary image.
    If the input image is a color image, first convert it to a grayscale image and then binarize it.
    :param src_img: Input image
    :param threshold: Threshold
    :return:Output binary image
    """
    ret = src_img.shape
    rows, cols = ret[:2]
    dst_img = np.zeros([rows, cols], dtype=np.float16)
    if len(ret) == 3 and ret[2] == 3:
        dst_img = rgb2gray(src_img)
    else:
        dst_img = src_img
    dst_img = dst_img / 255
    for i in range(rows):
        for j in range(cols):
            if dst_img[i, j] >= threshold/255:
                dst_img[i, j] = 1
            else:
                dst_img[i, j] = 0
    return dst_img


def main(img_path):
    img = cv.imread(img_path, -1)
    img_gray = rgb2gray(img)
    img_bin = gray2binary(img, 128)
    cv.imshow('src_img', img)
    cv.imshow('gray_img', img_gray)
    cv.imshow('binary image', img_bin)
    cv.waitKey(0)


if __name__ == '__main__':
    image_path = 'lenna.png'
    main(image_path)


