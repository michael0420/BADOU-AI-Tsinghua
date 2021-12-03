import cv2
import numpy as np


def binary_inter(img, new_dim):
    """
    biliary interpolation for the origin image
    :param img: image of a picture
    :param new_dim: a tuple of new image size
    :return: an interpolated image
    """
    # get original size of the image h X w X c
    src_h, src_w, channels = img.shape
    # get the destiny image size
    dst_h, dst_w = new_dim[0], new_dim[1]
    # if the size of source img and destiny img are same, return the copy of the source img
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros([dst_h, dst_w, channels], img.dtype)
    # x coordinate is the scale of the columns, y coordinate is the rows. Keep them float for calculation
    unit_x, unit_y = float(src_w) / dst_w, float(src_h) / dst_h
    for c in range(channels):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # re-balance to the center of the picture
                src_x, src_y = (dst_x + 0.5) * unit_x - 0.5, (dst_y + 0.5) * unit_y - 0.5

                # find src_x0, src_x1, src_y0, and src_y1 for further calculation
                src_x0, src_y0 = int(np.floor(src_x)), int(np.floor(src_y))
                src_x1, src_y1 = min(src_x0 + 1, src_w - 1), min(src_y0 + 1, src_h - 1)

                # switch back the y for height and x for width and the channels, calculate the interpolation
                r0 = (src_x1 - src_x) * img[src_y0, src_x0, c] + (src_x - src_x0) * img[src_y0, src_x1, c]
                r1 = (src_x1 - src_x) * img[src_y1, src_x0, c] + (src_x - src_x0) * img[src_y1, src_x1, c]
                dst_img[dst_y, dst_x, c] = int((src_y1 - src_y) * r0 + (src_y - src_y0) * r1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    new_pic = binary_inter(img, (700, 700))
    print("image show new_pic: %s" % new_pic)
    cv2.imshow('new_pic show', new_pic)
    cv2.waitKey(0)
