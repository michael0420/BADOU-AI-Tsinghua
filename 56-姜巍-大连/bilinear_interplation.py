import numpy as np
import cv2
import math

'''
let's do some bilinear interpolations ~
'''
# import time
# def timer(bi_func):
#     def wrapper(a, b):
#         start = time.time()
#         bi_func(a, b)
#         end = time.time()
#         return end - start
#
#     return wrapper
#
# By using decorator above, we'll know it takes 13.496195077896118 seconds when dst_h, dst_w = 800, 800


def bilinear_interpolation(image_src, dst_image_hw):
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
    # By using "for-loop" to do coordinate transformation, bilinear interpolation calculation and filling the values.
    for z in range(channels):
        for yd in range(dst_h):
            # do coordinate transformation on dst_image
            # to even the y coordinates of geometrical centers of two image(mapping image & source image)
            ym = (yd + 0.5) / ratio_h - 0.5
            # You MUST take all circumstances into consideration
            # There are totally 9 circumstances includes 3 ratio_h * 3 ratio_w
            # 3 ratio_h:(>1, <1, ==1);
            if ratio_h > 1:
                if ym <= 0:
                    ys_dn = 0
                elif ym >= (src_h - 1):
                    ys_dn = src_h - 2
                else:
                    ys_dn = math.floor(ym)
            elif ratio_h < 1:
                ys_dn = math.floor(ym)
            else:
                ys_dn = min(yd, src_h - 2)
            ys_up = ys_dn + 1
            for xd in range(dst_w):
                # to even the x coordinates of geometrical centers of two image(mapping image & source image)
                xm = (xd + 0.5) / ratio_w - 0.5
                # 3 ratio_w:(>1, <1, ==1)
                if ratio_w > 1:
                    if xm <= 0:
                        xs_lf = 0
                    elif xm >= (src_w - 1):
                        xs_lf = src_w - 2
                    else:
                        xs_lf = math.floor(xm)
                elif ratio_w < 1:
                    xs_lf = math.floor(xm)
                else:
                    xs_lf = min(xd, src_w - 2)
                xs_rt = xs_lf + 1
                # now do bilinear interpolation calculation by using "4-term form" interpolation formula.
                dst_image[yd, xd, z] = image_src[ys_up, xs_rt, z] * (xm - xs_lf) * (ym - ys_dn) \
                                       + image_src[ys_up, xs_lf, z] * (xs_rt - xm) * (ym - ys_dn) \
                                       + image_src[ys_dn, xs_rt, z] * (xm - xs_lf) * (ys_up - ym) \
                                       + image_src[ys_dn, xs_lf, z] * (xs_rt - xm) * (ys_up - ym)
    # destination image accomplished !!!
    return dst_image


if __name__ == '__main__':
    dst_img_hw = (int(input('pls give the width of destination image:')),
                  int(input('\n pls give the height of destination image:')))
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, dst_img_hw)
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()
