import numpy as np
import cv2
import math

'''
let's do some nearest interpolations ~
'''


def nearest_interpolation(image_src, dst_image_hw):
    # Get height, width & channel number info. of source image.
    src_h, src_w, channels = image_src.shape
    # Get height, width info. of destination image.
    dst_h, dst_w = dst_image_hw[1], dst_image_hw[0]
    print(channels)
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    # in case one type exactly the dst_h & dst_w same with the src_h & src_w
    if src_h == dst_h and src_w == dst_w:
        return image_src.copy()
    # And we'll know the ratio by calculating
    ratio_h = dst_h / src_h
    ratio_w = dst_w / src_w
    # Build a blank(0s-filled) destination image with 3 dimensions(height,width,nums of channel);
    # Its height = dst_h, width = dst_w and same numbers of channels as the source image.
    dst_image = np.zeros((dst_h, dst_w, channels), dtype=np.uint8)

    # By using "for-loop" to do coordinate transformation, nearest interpolation calculation and filling the values.
    for z in range(channels):
        for yd in range(dst_h):
            ym = (yd + 0.5) / ratio_h - 0.5
            bridge_y = lambda y: max(math.floor(y), 0) if (abs(y - math.floor(y)) <= 0.5) else min(math.ceil(y),
                                                                                                   src_h - 1)
            yt = bridge_y(ym)
            for xd in range(dst_w):
                # 1st, do coordinate transformation on dst_image
                # to even the coordinates of geometrical centers of two image(mapping image & source image)

                xm = (xd + 0.5) / ratio_w - 0.5
                # 2nd, in the source image, find the nearest coordinate relative to (ym , xm)
                # using "lambda function" seems to be brief
                # the following formula can make sure yt(or xt) ranges from 0 to (src_h - 1)

                bridge_x = lambda x: max(math.floor(xm), 0) if (abs(xm - math.floor(xm)) <= 0.5) else min(math.ceil(xm),
                                                                                                          src_w - 1)
                xt = bridge_x(xm)
                # 3rd, do nearest interpolation calculation.
                dst_image[yd, xd, z] = image_src[yt, xt, z]
    # destination image get
    return dst_image


if __name__ == '__main__':
    dst_img_hw = (int(input('pls give the width of destination image:')),
                  int(input('\n pls give the height of destination image:')))
    img = cv2.imread('lenna.png')
    dst = nearest_interpolation(img, dst_img_hw)
    cv2.imshow('nearest interp', dst)
    cv2.waitKey()
