
# bilinear interpolation implementation

# import library
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# bilinear interpolation function
def bilinear_interpolation(src_img, dst_height, dst_width):

    # size of source image
    src_h = src_img.shape[0]
    src_w = src_img.shape[1]
    channel = src_img.shape[2]

    # size of
    dst_h = dst_height
    dst_w = dst_width

    # output size of source image and destination image
    print ("src_h, src_w = ", src_h, src_w)
    print ("dst_h, dst_w = ", dst_h, dst_w)

    # size change check
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # create a new image
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)

    # scale calculation
    scale_x = float(src_w) / dst_w
    scale_y = float(src_h) / dst_h

    # channel loop
    for i in range(3):

        # row index loop
        for dst_y in range(dst_h):

            # col index loop
            for dst_x in range(dst_w):
 
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x-0.5
                src_y = (dst_y + 0.5) * scale_y-0.5
 
                # find the coordinates of the points which will be used to compute the interpolation
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1 ,src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)
 
                # calculate the interpolation
                temp0 = (src_x1 - src_x) * img[src_y0,src_x0,i] + (src_x - src_x0) * img[src_y0,src_x1,i]
                temp1 = (src_x1 - src_x) * img[src_y1,src_x0,i] + (src_x - src_x0) * img[src_y1,src_x1,i]
                dst_img[dst_y,dst_x,i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
 
    return dst_img

if __name__ == '__main__':
    img = cv2.imread('AWACS.jpeg')
    des_height = 600
    des_width =900
    dst = bilinear_interpolation(img,des_height,des_width)
    cv2.imshow('bilinear interp',dst)
    cv2.waitKey(20000)
