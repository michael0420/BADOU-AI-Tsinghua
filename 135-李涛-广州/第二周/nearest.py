import cv2
import numpy as np


def nearest_insert(image):
    height, width, channels = image.shape
    image_shape = (1024, 1024, channels)
    empty_iamge = np.zeros(image_shape, dtype=np.uint8)
    scale_height = image_shape[0] / height
    scale_width = image_shape[1] / width

    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            x = int(i / scale_height)
            y = int(j / scale_width)
            empty_iamge[i, j] = image[x, y]

    return empty_iamge

if __name__ == '__main__':
    image = cv2.imread('lenna.png')
    new_image = nearest_insert(image)

    cv2.imshow('origin',image)
    cv2.imshow('new',new_image)
    cv2.waitKey(0)